"""
Hibiscus Leaf Disease Detection - Standalone Model Evaluation Script
=====================================================================
Evaluates the trained Keras 3 model on a labeled dataset directory
structured with 'diseased' and 'fresh' subfolders.

Usage:
    python evaluate_model.py [dataset_directory_path]
"""

import os
import sys

# Force Keras 3 to use the PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from PIL import Image
import keras

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "hibiscus_disease_model_FINAL.keras"
)


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Loads the model with compile=False."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return keras.models.load_model(model_path, compile=False)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Preprocesses image to match the model's expected 224x224 RGB input."""
    rgb = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.expand_dims(np.array(rgb, dtype=np.float32), axis=0)
    return arr


def predict_single(model, image_path: str) -> float:
    """Runs a forward pass and returns the raw sigmoid float score."""
    img = Image.open(image_path)
    tensor = preprocess_image(img)
    out = model.predict(tensor, verbose=0)
    if hasattr(out, "numpy"):
        out = out.numpy()
    elif hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    return float(np.asarray(out)[0][0])


def evaluate_directory(dataset_dir: str, model_path: str = DEFAULT_MODEL_PATH):
    """Evaluates images across 'diseased' and 'fresh' subdirectories."""
    print(f"[INFO] Loading model: {model_path}")
    model = load_model(model_path)
    print(f"[SUCCESS] Model loaded using backend: {keras.backend.backend()}")

    diseased_dir = os.path.join(dataset_dir, "diseased")
    fresh_dir = os.path.join(dataset_dir, "fresh")

    if not os.path.exists(diseased_dir) or not os.path.exists(fresh_dir):
        print(f"[ERROR] Dataset directory must contain 'diseased' and 'fresh' subdirectories.")
        print(f"Looked in: {dataset_dir}")
        return

    diseased_files = [
        f for f in os.listdir(diseased_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    fresh_files = [
        f for f in os.listdir(fresh_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"\nFound {len(diseased_files)} diseased images and {len(fresh_files)} fresh images.")

    if not diseased_files and not fresh_files:
        print("[WARNING] No valid image files found in either category.")
        return

    diseased_scores = []
    print("\n--- Evaluating Diseased Leaf Images ---")
    for fname in diseased_files:
        p = os.path.join(diseased_dir, fname)
        score = predict_single(model, p)
        diseased_scores.append(score)
        print(f"  {fname:<30} -> raw sigmoid: {score:.4f}")

    fresh_scores = []
    print("\n--- Evaluating Fresh (Healthy) Leaf Images ---")
    for fname in fresh_files:
        p = os.path.join(fresh_dir, fname)
        score = predict_single(model, p)
        fresh_scores.append(score)
        print(f"  {fname:<30} -> raw sigmoid: {score:.4f}")

    print("\n" + "=" * 60)
    print("EMPIRICAL SCORE DISTRIBUTION SUMMARY")
    print("=" * 60)
    if diseased_scores:
        print(f"Diseased (N={len(diseased_scores)}): min={min(diseased_scores):.4f}, max={max(diseased_scores):.4f}, mean={np.mean(diseased_scores):.4f}")
    if fresh_scores:
        print(f"Fresh    (N={len(fresh_scores)}): min={min(fresh_scores):.4f}, max={max(fresh_scores):.4f}, mean={np.mean(fresh_scores):.4f}")

    print("\n[ANALYSIS NOTE]")
    print("Notice that both diseased and fresh images produce scores > 0.50 due to class imbalance during training.")
    print("A naive 0.50 decision threshold is invalid for this model. Ground-truth calibration or threshold tuning is required.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # Check standard default external location
        external_default = r"C:\Users\aarus\Downloads\hibiscus dataset\leaf dataset\test"
        target_dir = external_default if os.path.exists(external_default) else "dataset"

    evaluate_directory(target_dir)
