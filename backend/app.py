"""
Hibiscus Leaf Disease Detection - Flask Application Entry Point
===============================================================
Exposes REST endpoints:
  - GET  /api/health   : Service, model readiness, and database status
  - POST /api/predict  : Leaf image classification inference & scan history persistence
  - GET  /api/history  : Retrieves recent leaf scan history from MySQL
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Explicitly load .env from the backend directory at startup with override=True
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Import predictor singleton (loads model with Keras 3 + PyTorch backend)
from predictor import predictor, ALLOWED_EXTENSIONS

# Import database module for scan history persistence
import database

# Import Gemini visual symptom analyzer
import gemini_analyzer


# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)

app = Flask(__name__)

import re
LOCAL_ORIGIN_REGEX = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")

# Configure CORS for local development (supports Vite / React dev & preview on localhost/127.0.0.1)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                LOCAL_ORIGIN_REGEX,
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:5174",
                "http://127.0.0.1:5174",
                "http://localhost:5175",
                "http://127.0.0.1:5175",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "https://hibsicusai.netlify.app",
                "https://hibiscusai.netlify.app"
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    }
)

# Limit maximum upload payload to 16MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Attempt non-blocking schema initialization if MySQL is already running
try:
    init_ok, init_msg = database.initialize_schema()
    if init_ok:
        app.logger.info("Database schema verified: %s", init_msg)
    else:
        app.logger.warning("Database schema initialization deferred: %s", init_msg)
except Exception as e:
    app.logger.warning("Database startup notice: %s", e)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Returns service health, model readiness, Gemini readiness, and database status without exposing secrets."""
    model_ready = predictor is not None and predictor.is_ready()
    db_connected, db_message = database.check_db_status()
    gemini_diag = gemini_analyzer.get_gemini_diagnostics()
    gemini_ready = gemini_diag["key_configured"]

    return jsonify({
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "gemini_configured": gemini_ready,
        "gemini_diagnostics": gemini_diag,
        "database_connected": db_connected,
        "database_status": db_message,
        "service": "Hibiscus Leaf Disease Detection ML API"
    }), (200 if model_ready else 503)




@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts an uploaded image file via multipart/form-data under key 'image'.
    Returns classification probability score, metadata, and persists record to MySQL.
    """
    # 1. Verify model is loaded
    if predictor is None or not predictor.is_ready():
        return jsonify({
            "success": False,
            "error": "ML model is not available or failed to initialize."
        }), 503

    # 2. Validate multipart form contains 'image' field
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Missing 'image' file in multipart form upload."
        }), 400

    file = request.files["image"]

    # 3. Validate non-empty filename
    if not file or file.filename.strip() == "":
        return jsonify({
            "success": False,
            "error": "No file selected for upload."
        }), 400

    # 4. Validate allowed file extension
    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file type '.{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        }), 400

    # 5. Validate image data integrity using Pillow
    try:
        image = Image.open(file.stream)
        image.verify()  # Verifies file header and image integrity
        file.stream.seek(0)
        image = Image.open(file.stream)  # Re-open after verify
    except Exception:
        return jsonify({
            "success": False,
            "error": "The uploaded file could not be parsed as a valid image."
        }), 400

    # 6. Execute model inference safely
    try:
        result = predictor.predict(image)
    except Exception as exc:
        app.logger.error("Inference failure: %s", exc)
        return jsonify({
            "success": False,
            "error": "An internal error occurred while processing leaf prediction."
        }), 500

    # 7. Persist prediction to MySQL scan_history table (graceful degradation if DB is offline)
    try:
        saved, scan_id, db_err = database.save_scan(
            filename=filename,
            predicted_class=result.get("prediction") or "Unknown",
            raw_score=result["raw_score"],
            threshold=result["threshold"],
            threshold_status=result["threshold_status"]
        )
        result["saved_to_db"] = saved
        result["scan_id"] = scan_id
        if not saved:
            result["db_status"] = "Scan could not be saved to MySQL (database offline or unavailable)."
        else:
            result["db_status"] = "Scan successfully recorded in MySQL history."
    except Exception as db_exc:
        app.logger.warning("Database persistence error: %s", db_exc)
        result["saved_to_db"] = False
        result["scan_id"] = None
        result["db_status"] = "Database persistence failed."

    return jsonify(result), 200


@app.route("/api/classify-symptom", methods=["POST"])
def classify_symptom():
    """
    Accepts an uploaded hibiscus leaf image file via multipart/form-data under key 'image'.
    Calls Gemini API to analyze visual symptom categories (Stage 2 Visual Analysis):
      - Healthy, Chewed, Dried, Leaf Spot, Yellowing, Other / Uncertain.
    Returns structured JSON with category, visible symptoms, explanation, and uncertainty flag.
    Preserves separation from the Keras binary ML model.
    """
    # 1. Validate multipart form contains 'image' field
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Missing 'image' file in multipart form upload."
        }), 400

    file = request.files["image"]

    # 2. Validate non-empty filename
    if not file or file.filename.strip() == "":
        return jsonify({
            "success": False,
            "error": "No file selected for upload."
        }), 400

    # 3. Validate allowed file extension
    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file type '.{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        }), 400

    # 4. Validate image data integrity using Pillow
    try:
        image = Image.open(file.stream)
        image.verify()  # Verifies file header and image integrity
        file.stream.seek(0)
        image_bytes = file.stream.read()
        file.stream.seek(0)
    except Exception:
        return jsonify({
            "success": False,
            "error": "The uploaded file could not be parsed as a valid image."
        }), 400

    # Map file extension to MIME type
    mime_type_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp"
    }
    mime_type = mime_type_map.get(ext, "image/jpeg")

    # 5. Execute Gemini visual symptom classification
    result = gemini_analyzer.analyze_leaf_symptoms(image_bytes=image_bytes, mime_type=mime_type)

    if not result.get("success"):
        status_code = 503 if result.get("code") == "MISSING_API_KEY" else 500
        return jsonify(result), status_code

    return jsonify(result), 200


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Retrieves recent scan history records from MySQL, newest first.
    Returns HTTP 503 with helpful message if database is currently offline.
    """
    limit = request.args.get("limit", 20, type=int)
    success, records, err_msg = database.get_scan_history(limit=limit)

    if not success:
        return jsonify({
            "success": False,
            "error": "Database service is offline or inaccessible.",
            "details": err_msg,
            "records": []
        }), 503

    return jsonify({
        "success": True,
        "count": len(records),
        "records": records
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handles uploads exceeding MAX_CONTENT_LENGTH."""
    return jsonify({
        "success": False,
        "error": "Uploaded image file exceeds the 16MB maximum file size limit."
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found."
    }), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
