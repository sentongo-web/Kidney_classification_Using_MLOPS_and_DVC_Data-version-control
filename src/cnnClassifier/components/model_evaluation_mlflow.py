import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dagshub
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


def compute_expected_calibration_error(
    confidences: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes the Expected Calibration Error (ECE) for a validation cohort.

    The confidence axis [0, 1] is segmented into `n_bins` uniformly spaced
    bins. Within each bin, the absolute gap between mean predicted confidence
    and empirical (observed) accuracy is weighted by the bin's share of the
    total validation population, then summed across all bins:

        ECE = sum_b ( |bin_count| / N ) * | acc(bin_b) - conf(bin_b) |

    A well-calibrated clinical model should report an ECE close to zero,
    meaning its softmax confidence can be trusted as a genuine probability
    of correctness rather than an overconfident or underconfident score.

    Args:
        confidences: 1-D array of the top-class softmax confidence per sample.
        predicted_labels: 1-D array of the argmax predicted class per sample.
        true_labels: 1-D array of the ground-truth class per sample.
        n_bins: Number of uniformly spaced confidence bins. Defaults to 10.

    Returns:
        float: The scalar Expected Calibration Error, in [0, 1].
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correctness = (predicted_labels == true_labels).astype(np.float64)
    total_samples = confidences.shape[0]

    if total_samples == 0:
        logger.warning("ECE computation received an empty validation cohort; returning 0.0")
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_edges[:-1], bin_edges[1:]):
        if bin_lower == 0.0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_weight = np.sum(in_bin) / total_samples
        if bin_weight == 0.0:
            continue

        bin_accuracy = float(np.mean(correctness[in_bin]))
        bin_confidence = float(np.mean(confidences[in_bin]))
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


class Evaluation:
    """Runs statistical evaluation of a trained kidney classifier, covering
    both raw discriminative performance (loss, accuracy) and probability
    calibration (Expected Calibration Error), then logs everything to
    MLflow via DagsHub for full experiment traceability.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.model: tf.keras.Model
        self.score: list[float]
        self.ece: float = 0.0

    def _valid_generator(self) -> None:
        """Builds the held-out validation data generator (30% split, unshuffled
        so predictions can be aligned index-for-index with ground-truth labels).
        """
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.30)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        logger.info(
            f"Validation generator initialised with {self.valid_generator.samples} samples "
            f"across {self.valid_generator.num_classes} classes"
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Loads a registered Keras model artifact from disk.

        Args:
            path (Path): Path to the .h5 / .keras model artifact.

        Returns:
            tf.keras.Model: The deserialized model, ready for inference.
        """
        return tf.keras.models.load_model(path)

    def _compute_calibration(self) -> float:
        """Runs a full inference sweep over the validation cohort and derives
        the Expected Calibration Error from the resulting confidence/accuracy
        distribution.

        Returns:
            float: The Expected Calibration Error (ECE) for this model.
        """
        self.valid_generator.reset()
        probabilities = self.model.predict(self.valid_generator, verbose=0)
        true_labels = self.valid_generator.classes[: probabilities.shape[0]]

        confidences = np.max(probabilities, axis=1)
        predicted_labels = np.argmax(probabilities, axis=1)

        ece = compute_expected_calibration_error(
            confidences=confidences,
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            n_bins=10,
        )
        logger.info(f"Expected Calibration Error (ECE) computed: {ece:.6f}")
        return ece

    def evaluation(self) -> None:
        """Executes the full evaluation routine: loads the model, builds the
        validation cohort, computes cross-entropy loss/accuracy, derives the
        calibration statistic, and persists results to scores.json.
        """
        try:
            self.model = self.load_model(self.config.path_of_model)
            self._valid_generator()
            self.score = self.model.evaluate(self.valid_generator)
            self.ece = self._compute_calibration()
            self.save_score()
        except Exception as e:
            logger.exception(f"Model evaluation failed: {e}")
            raise e

    def save_score(self) -> None:
        """Persists the loss, accuracy, and ECE metrics to a local scores.json
        artifact for DVC metric tracking.
        """
        scores: dict[str, Any] = {
            "loss": self.score[0],
            "accuracy": self.score[1],
            "ECE": self.ece,
        }
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self) -> None:
        """Logs parameters, metrics (including ECE), and the model artifact
        to the remote MLflow tracking workspace hosted on DagsHub.
        """
        try:
            dagshub.init(
                repo_owner="sentongo-web",
                repo_name="Kidney_classification_Using_MLOPS_and_DVC_Data-version-control",
                mlflow=True
            )
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics({
                    "loss": self.score[0],
                    "accuracy": self.score[1],
                    "ECE": self.ece,
                })

                if tracking_url_type_store != "file":
                    mlflow.tensorflow.log_model(self.model, "model", registered_model_name="VGG16Model")
                else:
                    mlflow.tensorflow.log_model(self.model, "model")

            logger.info("Evaluation metrics and model artifact logged to MLflow successfully")
        except Exception as e:
            logger.exception(f"Failed to log evaluation run to MLflow: {e}")
            raise e
