from unittest.mock import MagicMock

import numpy as np

from cnnClassifier.pipeline.prediction import PredictionPipeline


class _ConstantModel:
    """A stand-in Keras model that always returns a fixed softmax output,
    used to deterministically exercise each PredictionPipeline branch
    without needing a real trained network.
    """

    def __init__(self, probabilities):
        self._probabilities = np.array([probabilities], dtype=np.float32)

    def predict(self, _array):
        return self._probabilities


def test_predict_normal_does_not_trigger_xai(sample_image_path):
    model = _ConstantModel([0.9, 0.1])  # index 0 = Normal, high confidence
    xai_engine = MagicMock()

    pipeline = PredictionPipeline(sample_image_path, model=model, xai_engine=xai_engine)
    result = pipeline.predict()

    assert result[0]["image"] == "Normal"
    assert result[0]["gradcam_path"] == ""
    assert result[0]["counterfactual_path"] == ""
    xai_engine.generate_gradcam.assert_not_called()
    xai_engine.generate_counterfactual.assert_not_called()


def test_predict_tumor_triggers_xai_evidence_generation(sample_image_path):
    model = _ConstantModel([0.05, 0.95])  # index 1 = Tumor, high confidence
    xai_engine = MagicMock()
    xai_engine.generate_gradcam.return_value = "/artifacts/gradcam.png"
    xai_engine.generate_counterfactual.return_value = "/artifacts/counterfactual.png"

    pipeline = PredictionPipeline(sample_image_path, model=model, xai_engine=xai_engine)
    result = pipeline.predict()

    assert result[0]["image"] == "Tumor"
    assert result[0]["gradcam_path"] == "/artifacts/gradcam.png"
    assert result[0]["counterfactual_path"] == "/artifacts/counterfactual.png"
    xai_engine.generate_gradcam.assert_called_once()
    xai_engine.generate_counterfactual.assert_called_once()


def test_predict_low_confidence_is_flagged_invalid_and_skips_xai(sample_image_path):
    model = _ConstantModel([0.55, 0.45])
    xai_engine = MagicMock()

    pipeline = PredictionPipeline(sample_image_path, model=model, xai_engine=xai_engine)
    result = pipeline.predict()

    assert result[0]["image"] == "InvalidImage"
    assert result[0]["gradcam_path"] == ""
    assert result[0]["counterfactual_path"] == ""
    xai_engine.generate_gradcam.assert_not_called()


def test_xai_failure_is_swallowed_and_prediction_still_succeeds(sample_image_path):
    """A clinician must always receive a classification, even if the
    explanatory evidence pipeline itself breaks — an image upload should
    never 500 just because Grad-CAM/counterfactual generation failed.
    """
    model = _ConstantModel([0.02, 0.98])
    xai_engine = MagicMock()
    xai_engine.generate_gradcam.side_effect = RuntimeError("simulated XAI failure")

    pipeline = PredictionPipeline(sample_image_path, model=model, xai_engine=xai_engine)
    result = pipeline.predict()  # must not raise

    assert result[0]["image"] == "Tumor"
    assert result[0]["gradcam_path"] == ""
    assert result[0]["counterfactual_path"] == ""
