import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Minimum softmax confidence required to trust a prediction.
# Below this threshold the image is likely not a kidney CT scan.
CONFIDENCE_THRESHOLD = 0.80


class PredictionPipeline:
    def __init__(self, filename, model=None):
        self.filename = filename
        self._model = model

    def predict(self):
        # Use pre-loaded model if provided, otherwise load from disk
        if self._model is not None:
            model = self._model
        else:
            keras_path = os.path.join("artifacts", "training", "model.keras")
            h5_path    = os.path.join("artifacts", "training", "model.h5")
            model_path = keras_path if os.path.isfile(keras_path) else h5_path
            model = load_model(model_path)

        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        confidence  = float(np.max(predictions))
        class_idx   = int(np.argmax(predictions, axis=1)[0])

        if confidence < CONFIDENCE_THRESHOLD:
            return [{"image": "InvalidImage", "confidence": round(confidence, 4)}]

        return [{"image": "Tumor" if class_idx == 1 else "Normal",
                 "confidence": round(confidence, 4)}]
