import os
import sys

# 1. Force Keras 3 to use PyTorch BEFORE importing keras
os.environ["KERAS_BACKEND"] = "torch"

try:
    import torch
    print(f"[OK] PyTorch imported successfully: version {torch.__version__}")
except Exception as e:
    print(f"[ERROR] Failed to import PyTorch: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"[OK] NumPy imported successfully: version {np.__version__}")
except Exception as e:
    print(f"[ERROR] Failed to import NumPy: {e}")
    sys.exit(1)

try:
    import h5py
    print(f"[OK] h5py imported successfully: version {h5py.__version__}")
except Exception as e:
    print(f"[ERROR] Failed to import h5py: {e}")
    sys.exit(1)

try:
    import keras
    print(f"[OK] Keras imported successfully: version {keras.__version__}")
    print(f"[OK] Active Keras Backend: {keras.backend.backend()}")
except Exception as e:
    print(f"[ERROR] Failed to import Keras: {e}")
    sys.exit(1)

if keras.backend.backend() != "torch":
    print(f"[ERROR] Expected Keras backend 'torch', but got '{keras.backend.backend()}'")
    sys.exit(1)

# 2. Model path resolution (relative to project root: backend/models/...)
model_path = os.path.join("backend", "models", "hibiscus_disease_model_FINAL.keras")
if not os.path.exists(model_path):
    if os.path.exists(os.path.join("models", "hibiscus_disease_model_FINAL.keras")):
        model_path = os.path.join("models", "hibiscus_disease_model_FINAL.keras")
    else:
        print(f"[ERROR] Model file not found at: {model_path}")
        sys.exit(1)

print(f"\n[INFO] Loading model from: {model_path} (compile=False)...")
try:
    model = keras.models.load_model(model_path, compile=False)
    print("[SUCCESS] Model loaded successfully with PyTorch backend!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Print input and output shapes
input_shape = getattr(model, "input_shape", None)
output_shape = getattr(model, "output_shape", None)
print(f"[INFO] Model input shape: {input_shape}")
print(f"[INFO] Model output shape: {output_shape}")

# 4. Run dummy forward pass with input shape (1, 224, 224, 3)
print("\n[INFO] Running dummy forward pass with shape (1, 224, 224, 3)...")
try:
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
    output = model.predict(dummy_input)
    
    if hasattr(output, "numpy"):
        output = output.numpy()
    elif hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    output = np.asarray(output)

    print(f"[SUCCESS] Dummy forward pass completed!")
    print(f"[INFO] Actual output shape: {output.shape}")
    print(f"[INFO] Output raw array: {output}")

    if output.shape != (1, 1):
        print(f"[WARNING] Expected output shape (1, 1), but got {output.shape}")
    else:
        print("[OK] Output shape verified: (1, 1)")

    pred_prob = float(output[0][0])
    if np.isnan(pred_prob) or np.isinf(pred_prob):
        print(f"[ERROR] Output probability is not finite: {pred_prob}")
        sys.exit(1)

    if 0.0 <= pred_prob <= 1.0:
        print(f"[OK] Output probability verified: {pred_prob:.4f} (valid range [0, 1])")
    else:
        print(f"[WARNING] Output probability {pred_prob:.4f} is outside [0, 1] range")

    print("\n[TEST RESULT] Keras 3 + PyTorch backend compatibility test PASSED!")

except Exception as e:
    print(f"[ERROR] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
