import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from cnnClassifier import logger
from cnnClassifier.components.explainability_engine import ExplainabilityEngine
from cnnClassifier.config.configuration import ConfigurationManager

# Minimum softmax confidence required to trust a prediction.
# Below this threshold the image is likely not a kidney CT scan.
CONFIDENCE_THRESHOLD = 0.80


class PredictionPipeline:
    def __init__(
        self,
        filename: str,
        model: Any = None,
        xai_engine: Optional[ExplainabilityEngine] = None,
    ) -> None:
        self.filename = filename
        self._model = model
        self._xai_engine = xai_engine

    def _get_xai_engine(self) -> ExplainabilityEngine:
        """Lazily resolves the ExplainabilityEngine, reusing an
        injected instance (e.g. one built once at Flask startup) when
        available, otherwise constructing one from the on-disk config.
        """
        if self._xai_engine is None:
            xai_config = ConfigurationManager().get_explainability_config()
            self._xai_engine = ExplainabilityEngine(config=xai_config)
        return self._xai_engine

    def predict(self) -> list[dict[str, Any]]:
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
            return [{
                "image": "InvalidImage",
                "confidence": round(confidence, 4),
                "gradcam_path": "",
                "counterfactual_path": "",
            }]

        prediction_label = "Tumor" if class_idx == 1 else "Normal"
        gradcam_path = ""
        counterfactual_path = ""

        # Only pathological findings warrant the explanatory evidence trail —
        # a "Normal" reading has no tumor region to localize or counterfact.
        if prediction_label == "Tumor":
            try:
                engine = self._get_xai_engine()
                sample_id = Path(self.filename).stem
                gradcam_path = engine.generate_gradcam(
                    model=model, image_path=self.filename, output_name=sample_id
                )
                counterfactual_path = engine.generate_counterfactual(
                    model=model, image_path=self.filename, output_name=sample_id
                )
            except Exception as e:
                logger.exception(f"XAI evidence generation failed for '{self.filename}': {e}")

        return [{
            "image": prediction_label,
            "confidence": round(confidence, 4),
            "gradcam_path": gradcam_path,
            "counterfactual_path": counterfactual_path,
        }]
