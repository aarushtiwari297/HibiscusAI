"""
Unit & Integration Tests for HibiscusAI Backend API Endpoints
=============================================================
Tests:
1. GET  /api/health            : Verifies service, model loading, and gemini status
2. POST /api/predict           : Verifies existing Keras model binary inference contract
3. POST /api/classify-symptom  : Verifies image validation, error handling, and Gemini integration
"""

import io
import json
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

import app as flask_app
import gemini_analyzer


class BackendEndpointsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    def _create_sample_image(self, format="JPEG", size=(224, 224), color=(34, 139, 34)):
        """Generates a valid in-memory PIL image byte stream."""
        img = Image.new("RGB", size, color=color)
        buf = io.BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        return buf

    # -------------------------------------------------------------------------
    # 1. Health Check Endpoint Tests
    # -------------------------------------------------------------------------
    def test_health_check_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("gemini_configured", data)
        self.assertTrue(data["model_loaded"])

    # -------------------------------------------------------------------------
    # 2. Existing Binary Predict Endpoint Tests (Contract Preservation)
    # -------------------------------------------------------------------------
    def test_predict_missing_image_file(self):
        res = self.client.post("/api/predict", data={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Missing 'image'", data["error"])

    def test_predict_invalid_extension(self):
        data = {
            "image": (io.BytesIO(b"fake txt content"), "test.txt")
        }
        res = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        res_data = res.get_json()
        self.assertFalse(res_data["success"])
        self.assertIn("Unsupported file type", res_data["error"])

    def test_predict_valid_image_inference(self):
        img_buf = self._create_sample_image()
        data = {
            "image": (img_buf, "test_leaf.jpg")
        }
        res = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        res_data = res.get_json()
        self.assertTrue(res_data["success"])
        self.assertIn("prediction", res_data)
        self.assertIn(res_data["prediction"], ["Fresh", "Diseased"])
        self.assertIn("raw_score", res_data)
        self.assertIn("threshold", res_data)
        self.assertEqual(res_data["threshold"], 0.85)

    # -------------------------------------------------------------------------
    # 3. Gemini Visual Symptom Endpoint Tests
    # -------------------------------------------------------------------------
    def test_classify_symptom_missing_image(self):
        res = self.client.post("/api/classify-symptom", data={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])

    def test_classify_symptom_corrupted_image(self):
        data = {
            "image": (io.BytesIO(b"not an image header"), "broken.jpg")
        }
        res = self.client.post("/api/classify-symptom", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        res_data = res.get_json()
        self.assertFalse(res_data["success"])
        self.assertIn("could not be parsed as a valid image", res_data["error"])

    def test_classify_symptom_missing_api_key(self):
        """When GEMINI_API_KEY is not in env, endpoint returns 503 and graceful message."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            img_buf = self._create_sample_image()
            data = {"image": (img_buf, "sample_leaf.jpg")}
            res = self.client.post("/api/classify-symptom", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 503)
            res_data = res.get_json()
            self.assertFalse(res_data["success"])
            self.assertEqual(res_data.get("code"), "MISSING_API_KEY")

    def test_classify_symptom_mocked_success(self):
        """Simulates successful Gemini model classification with structured output."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "category": "Leaf Spot",
            "visible_symptoms": ["Discrete circular necrotic lesions", "Yellow halo around spots"],
            "explanation": "Visible necrotic spotting on the adaxial leaf surface with surrounding chlorotic rings.",
            "uncertain": False,
            "confidence": "high"
        })

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_dummy_key_123"}), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            img_buf = self._create_sample_image()
            data = {"image": (img_buf, "leaf_with_spots.jpg")}
            res = self.client.post("/api/classify-symptom", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 200)
            res_data = res.get_json()
            self.assertTrue(res_data["success"])
            self.assertEqual(res_data["category"], "Leaf Spot")
            self.assertFalse(res_data["uncertain"])
            self.assertEqual(len(res_data["visible_symptoms"]), 2)
            self.assertIn("disclaimer", res_data)

    def test_classify_symptom_unsupported_category_coercion(self):
        """If Gemini returns an unsupported category, verify it is coerced to Other / Uncertain."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "category": "Alien Invertebrate Attack",
            "visible_symptoms": ["Unusual marks"],
            "explanation": "Odd marks not in standard categories.",
            "uncertain": False,
            "confidence": "low"
        })

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_dummy_key_123"}), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            img_buf = self._create_sample_image()
            data = {"image": (img_buf, "leaf.png")}
            res = self.client.post("/api/classify-symptom", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 200)
            res_data = res.get_json()
            self.assertTrue(res_data["success"])
            self.assertEqual(res_data["category"], "Other / Uncertain")
            self.assertTrue(res_data["uncertain"])

    def test_classify_symptom_unsupported_extension(self):
        """Verifies that non-whitelisted image formats (e.g., .gif, .bmp) are rejected with HTTP 400."""
        data = {
            "image": (io.BytesIO(b"GIF89a..."), "leaf_scan.gif")
        }
        res = self.client.post("/api/classify-symptom", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        res_data = res.get_json()
        self.assertFalse(res_data["success"])
        self.assertIn("Unsupported file type", res_data["error"])

    def test_classify_symptom_api_rate_limit_error(self):
        """MOCK TEST: Simulates a 429 Resource Exhausted error from Google Gen AI SDK."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_dummy_key_123"}), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")
            mock_client_cls.return_value = mock_client

            img_buf = self._create_sample_image()
            data = {"image": (img_buf, "leaf.jpg")}
            res = self.client.post("/api/classify-symptom", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 500)
            res_data = res.get_json()
            self.assertFalse(res_data["success"])
            self.assertEqual(res_data.get("code"), "QUOTA_EXCEEDED")

    def test_classify_symptom_malformed_response(self):
        """MOCK TEST: Simulates Gemini returning empty or unparseable text."""
        mock_response = MagicMock()
        mock_response.text = ""

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_dummy_key_123"}), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            img_buf = self._create_sample_image()
            data = {"image": (img_buf, "leaf.jpg")}
            res = self.client.post("/api/classify-symptom", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 500)
            res_data = res.get_json()
            self.assertFalse(res_data["success"])

    def test_classify_symptom_uncertainty_flag(self):
        """MOCK TEST: Simulates Gemini identifying an ambiguous/blurry image and flagging uncertainty."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "category": "Other / Uncertain",
            "visible_symptoms": ["Severe motion blur", "Low illumination"],
            "explanation": "Image focus is insufficient to differentiate leaf pathology.",
            "uncertain": True,
            "confidence": "low"
        })

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_dummy_key_123"}), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            img_buf = self._create_sample_image()
            data = {"image": (img_buf, "blurry_leaf.jpg")}
            res = self.client.post("/api/classify-symptom", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 200)
            res_data = res.get_json()
            self.assertTrue(res_data["success"])
            self.assertEqual(res_data["category"], "Other / Uncertain")
            self.assertTrue(res_data["uncertain"])
            self.assertEqual(res_data["confidence"], "low")

    # -------------------------------------------------------------------------
    # 4. Single Guarded Live Gemini Test (Runs ONLY if GEMINI_API_KEY is configured)
    # -------------------------------------------------------------------------
    def test_live_gemini_classification_single_call(self):
        """
        LIVE TEST: Makes a single live call to Gemini API using the locally configured key.
        Guarded: Runs only when a valid GEMINI_API_KEY is present in the environment;
        skips if not configured to prevent failures in environments without keys.
        """
        if not gemini_analyzer.is_gemini_configured():
            self.skipTest("GEMINI_API_KEY is not configured; skipping live API test.")

        img_buf = self._create_sample_image(format="JPEG", size=(224, 224), color=(46, 139, 87))
        data = {"image": (img_buf, "live_audit_leaf.jpg")}

        res = self.client.post("/api/classify-symptom", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        res_data = res.get_json()

        self.assertTrue(res_data["success"])
        self.assertIn(res_data["category"], gemini_analyzer.SUPPORTED_CATEGORIES)
        self.assertIsInstance(res_data["visible_symptoms"], list)
        self.assertIsInstance(res_data["explanation"], str)
        self.assertIsInstance(res_data["uncertain"], bool)
        self.assertIn("disclaimer", res_data)


if __name__ == "__main__":
    unittest.main()

