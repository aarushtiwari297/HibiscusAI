"""
Hibiscus Leaf Disease Detection - ML Predictor Module
======================================================
Loads the trained Keras 3 model using the PyTorch CPU backend.
Handles image validation, RGB conversion, 224x224 resizing,
and batch tensor creation.
"""

import os
import sys

# 1. Force Keras 3 to use the PyTorch backend before any keras imports
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from PIL import Image
import keras

# ==============================================================================
# PROVISIONAL THRESHOLD & CLASS MAPPING CONFIGURATION
# ==============================================================================
# The model outputs a single scalar sigmoid value between 0.0 and 1.0.
#
# EMPIRICAL PROVISIONAL THRESHOLD:
# In an empirical evaluation of 18 test images from the dataset split, diseased
# leaves produced scores in [0.53, 0.81] (mean ~0.72) and fresh leaves produced
# scores in [0.97, 0.99] (mean ~0.98), likely due to a 3:1 training class imbalance.
# A decision threshold of 0.85 separates these test samples:
#   raw_score >= 0.85 -> "Fresh"
#   raw_score <  0.85 -> "Diseased"
#
# CAVEAT: This threshold is PROVISIONAL based on inspecting 18 test samples.
# It is NOT scientifically validated across broad field datasets, nor is it
# derived from authoritative training-pipeline code or training logs.
# ==============================================================================
DISEASE_THRESHOLD = 0.85
THRESHOLD_STATUS = "provisional"
CLASS_MAPPING_VERIFIED = False  # False: authoritative training artifact verification is absent

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MODEL_FILENAME = "hibiscus_disease_model_FINAL.keras"


class HibiscusPredictor:
    """Manages model loading and inference lifecycle."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            # Resolve model path relative to this script directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", MODEL_FILENAME)
        
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads Keras model once with compile=False."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at expected location: {self.model_path}"
            )
        
        print(f"[INFO] Initializing Keras model using backend: {keras.backend.backend()}")
        print(f"[INFO] Loading model from: {self.model_path}")
        self.model = keras.models.load_model(self.model_path, compile=False)
        print("[SUCCESS] Model successfully loaded and ready for inference.")

    def is_ready(self) -> bool:
        """Returns True if the model is loaded."""
        return self.model is not None

    def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Converts image to RGB, resizes to 224x224, and formats as batch tensor.

        NOTE ON NORMALIZATION:
        Inspection of the model's architecture (config.json) confirms that the model
        graph embeds internal normalization layers:
          - Layer 2: TrueDivide (divides sequential tensor by 127.5)
          - Layer 3: Subtract (subtracts 1.0)
        This corresponds to MobileNetV2's [-1, 1] scaling from [0, 255] raw pixel inputs.
        Applying manual division by 255.0 beforehand would cause double-scaling.
        Therefore, we provide raw float32 pixel values in the range [0.0, 255.0].
        """
        # Ensure 3-channel RGB
        rgb_image = pil_image.convert("RGB")
        # Resize to model expected input size
        resized_image = rgb_image.resize((224, 224), Image.Resampling.BILINEAR)
        # Convert to numpy array in [0.0, 255.0] float32
        img_array = np.array(resized_image, dtype=np.float32)
        # Expand dims to create batch: shape (1, 224, 224, 3)
        batch_array = np.expand_dims(img_array, axis=0)
        return batch_array

    def predict(self, pil_image: Image.Image) -> dict:
        """
        Executes model prediction on a PIL Image and returns structured output.
        """
        if not self.is_ready():
            raise RuntimeError("Model is not initialized or ready for inference.")

        batch_tensor = self.preprocess_image(pil_image)
        raw_pred = self.model.predict(batch_tensor, verbose=0)

        # Convert tensor/array to standard Python float
        if hasattr(raw_pred, "numpy"):
            raw_pred = raw_pred.numpy()
        elif hasattr(raw_pred, "detach"):
            raw_pred = raw_pred.detach().cpu().numpy()
        
        raw_score = float(np.asarray(raw_pred)[0][0])

        # Apply provisional threshold (raw_score < 0.85 -> Diseased, raw_score >= 0.85 -> Fresh)
        if raw_score >= DISEASE_THRESHOLD:
            predicted_class = "Fresh"
        else:
            predicted_class = "Diseased"

        return {
            "success": True,
            "prediction": predicted_class,
            "raw_score": round(raw_score, 4),
            "threshold": DISEASE_THRESHOLD,
            "threshold_status": THRESHOLD_STATUS,
            "class_mapping_verified": CLASS_MAPPING_VERIFIED,
            "message": (
                f"Prediction '{predicted_class}' assigned based on provisional empirical decision "
                f"threshold ({DISEASE_THRESHOLD}). Note: Threshold is provisional based on 18 test "
                f"samples and does not represent authoritative training pipeline validation."
            )
        }


# Global singleton instance created on module load
try:
    predictor = HibiscusPredictor()
except Exception as e:
    print(f"[ERROR] Failed to initialize HibiscusPredictor: {e}", file=sys.stderr)
    predictor = None
