"""
One-time script: resave model.h5 as model.keras (Keras 3 native format).
This ensures the deployed model loads correctly regardless of minor Keras version
differences between training and serving environments.

Run once:
    python resave_model.py
"""

from tensorflow.keras.models import load_model

h5_path    = "artifacts/training/model.h5"
keras_path = "artifacts/training/model.keras"

print(f"Loading  {h5_path} ...")
model = load_model(h5_path)

print(f"Saving   {keras_path} ...")
model.save(keras_path)

print("Done. Deploy with:  python deploy_to_hf.py")
