from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

import tensorflow as tf

from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.explainability_engine import ExplainabilityEngine

STAGE_NAME = "Explainability stage"

SAMPLES_PER_CLASS = 3
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class ExplainabilityPipeline:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _collect_sample_images(training_data_dir: Path, samples_per_class: int) -> List[Tuple[str, str]]:
        """Collects a bounded number of representative sample images from
        each class subfolder of the validation dataset.

        Args:
            training_data_dir (Path): Root directory containing one
                subfolder per class (e.g. "Normal", "Tumor").
            samples_per_class (int): Maximum number of images to sample
                from each class subfolder.

        Returns:
            List[Tuple[str, str]]: (class_name, image_path) pairs.
        """
        samples: List[Tuple[str, str]] = []
        training_data_dir = Path(training_data_dir)

        if not training_data_dir.is_dir():
            logger.warning(f"Training data directory not found: {training_data_dir}")
            return samples

        for class_dir in sorted(p for p in training_data_dir.iterdir() if p.is_dir()):
            image_files = sorted(
                f for f in class_dir.iterdir()
                if f.suffix.lower() in VALID_IMAGE_EXTENSIONS
            )[:samples_per_class]
            for image_file in image_files:
                samples.append((class_dir.name, str(image_file)))

        return samples

    def main(self) -> None:
        """Loads the registered model artifact and the resolved
        explainability configuration, samples representative images from
        the validation folders, and runs both the Grad-CAM and gradient
        counterfactual algorithms across the sample set — regenerating
        the full visual-evidence artifact tree on every pipeline sweep.
        """
        config_manager = ConfigurationManager()
        xai_config = config_manager.get_explainability_config()
        evaluation_config = config_manager.get_evaluation_config()

        logger.info(f"Loading registered model artifact from: {evaluation_config.path_of_model}")
        model = tf.keras.models.load_model(evaluation_config.path_of_model)

        engine = ExplainabilityEngine(config=xai_config)

        samples = self._collect_sample_images(
            training_data_dir=Path(evaluation_config.training_data),
            samples_per_class=SAMPLES_PER_CLASS,
        )

        if not samples:
            logger.warning(
                "No sample images were found under the validation data directory; "
                "explainability artifacts will not be generated for this sweep."
            )
            return

        for class_name, image_path in samples:
            output_name = f"{class_name}_{Path(image_path).stem}"

            engine.generate_gradcam(model=model, image_path=image_path, output_name=output_name)

            if class_name.strip().lower() == "tumor":
                engine.generate_counterfactual(model=model, image_path=image_path, output_name=output_name)

        logger.info(f"Explainability sweep complete — processed {len(samples)} sample image(s)")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ExplainabilityPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
