from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import ExplainabilityConfig


class ExplainabilityEngine:
    """Embedded Explainable AI (XAI) research engine for the kidney CT scan
    classifier.

    This engine produces two complementary forms of post-hoc visual evidence
    that a clinician can inspect alongside a raw softmax prediction:

    1. Grad-CAM spatial attention maps, which reveal *where* in the CT scan
       the network concentrated its evidence when arriving at its top
       predicted class.
    2. Gradient-optimized counterfactual images, which reveal *what* the
       input tissue would need to look like for the network to instead
       classify it as the target ("Normal") class — a pixel-space
       illustration of the decision boundary the model has learned.

    Attributes:
        config (ExplainabilityConfig): Resolved artifact paths and the name
            of the terminal convolutional layer used for Grad-CAM.
    """

    NORMAL_CLASS_INDEX: int = 0
    TUMOR_CLASS_INDEX: int = 1
    OVERLAY_ALPHA: float = 0.4

    def __init__(self, config: ExplainabilityConfig) -> None:
        self.config = config

    @staticmethod
    def _get_input_size(model: tf.keras.Model) -> Tuple[int, int]:
        """Derives the (height, width) the model expects at its input layer.

        Args:
            model (tf.keras.Model): The trained classifier.

        Returns:
            Tuple[int, int]: (height, width) of the model's input tensor.
        """
        _, height, width, _ = model.input_shape
        return int(height), int(width)

    @staticmethod
    def _load_and_preprocess(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
        """Loads an image from disk and normalizes it into a model-ready tensor.

        Args:
            image_path (str): Path to the source CT scan image.
            target_size (Tuple[int, int]): (height, width) to resize to.

        Returns:
            np.ndarray: A float32 array of shape (1, H, W, 3), scaled to [0, 1].
        """
        img = keras_image.load_img(image_path, target_size=target_size)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array.astype(np.float32)

    def generate_gradcam(self, model: tf.keras.Model, image_path: str, output_name: str) -> str:
        """Computes a Grad-CAM spatial attention map relative to the model's
        top predicted class and writes a JET-colormap overlay to disk.

        The routine:
          1. Forwards the image through a sub-model exposing both the
             terminal convolutional feature maps (`target_layer_name`) and
             the final class probabilities.
          2. Differentiates the top predicted class score with respect to
             those feature maps to obtain per-channel importance weights.
          3. Collapses the weighted feature maps into a single 2-D heatmap,
             applies a ReLU filter to retain only positive attributions,
             and min-max normalizes the result.
          4. Resizes the heatmap to the original frame, colorizes it with a
             JET colormap, and alpha-blends it over the raw pixel intensities.

        Args:
            model (tf.keras.Model): The trained, loaded classifier.
            image_path (str): Path to the CT scan image to explain.
            output_name (str): Base filename (without extension) for the
                resulting artifact.

        Returns:
            str: Absolute/relative path to the saved Grad-CAM overlay PNG.

        Raises:
            FileNotFoundError: If the image at `image_path` cannot be read.
            Exception: Propagates any failure encountered during map
                generation, after logging the full traceback.
        """
        try:
            original_bgr = cv2.imread(str(image_path))
            if original_bgr is None:
                raise FileNotFoundError(f"Could not read image at path: {image_path}")

            height, width = self._get_input_size(model)
            original_bgr = cv2.resize(original_bgr, (width, height))
            input_tensor = self._load_and_preprocess(image_path, (height, width))

            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[model.get_layer(self.config.target_layer_name).output, model.output],
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_tensor)
                top_class_index = tf.argmax(predictions[0])
                top_class_score = predictions[:, top_class_index]

            gradients = tape.gradient(top_class_score, conv_outputs)
            if gradients is None:
                raise RuntimeError(
                    f"Gradient computation returned None for layer '{self.config.target_layer_name}'. "
                    "Verify the layer name exists on the supplied model."
                )

            pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
            heatmap = tf.maximum(heatmap, 0)  # ReLU filter: isolate positive spatial attributions

            max_value = tf.math.reduce_max(heatmap)
            heatmap = heatmap / max_value if max_value > 0 else heatmap
            heatmap = heatmap.numpy()

            heatmap_resized = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            superimposed = cv2.addWeighted(
                heatmap_color, self.OVERLAY_ALPHA, original_bgr, 1 - self.OVERLAY_ALPHA, 0
            )

            output_path = Path(self.config.gradcam_output_dir) / f"{output_name}_gradcam.png"
            cv2.imwrite(str(output_path), superimposed)
            logger.info(f"Grad-CAM attention map saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.exception(f"Grad-CAM generation failed for '{image_path}': {e}")
            raise e

    def generate_counterfactual(
        self,
        model: tf.keras.Model,
        image_path: str,
        output_name: str,
        steps: int = 50,
        lr: float = 0.05,
    ) -> str:
        """Synthesizes a gradient-optimized counterfactual: the minimal pixel
        reconfiguration required to push a positive ("Tumor") scan across the
        decision boundary into the "Normal" class distribution.

        The network's parameters are frozen; only the input pixel tensor is
        treated as a trainable variable. An SGD optimizer performs gradient
        descent on the negative log-likelihood of the target ("Normal")
        class, clipping pixel values back into the valid [0, 1] range after
        every step. The resulting image answers, for a clinician: "if this
        tissue region were healthy, this is how its pixel density topology
        would be configured."

        Args:
            model (tf.keras.Model): The trained, loaded classifier. Its
                weights are frozen for the duration of this routine.
            image_path (str): Path to the source ("Tumor") CT scan image.
            output_name (str): Base filename (without extension) for the
                resulting artifact.
            steps (int): Number of gradient-descent iterations. Defaults to 50.
            lr (float): Learning rate for the pixel-space SGD optimizer.
                Defaults to 0.05.

        Returns:
            str: Path to the saved counterfactual simulation PNG.

        Raises:
            Exception: Propagates any failure encountered during
                optimization, after logging the full traceback.
        """
        try:
            model.trainable = False  # freeze network parameters; only pixels are optimized

            height, width = self._get_input_size(model)
            input_array = self._load_and_preprocess(image_path, (height, width))
            image_var = tf.Variable(input_array, trainable=True, dtype=tf.float32)
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

            for step in range(steps):
                with tf.GradientTape() as tape:
                    tape.watch(image_var)
                    predictions = model(image_var, training=False)
                    target_probability = predictions[:, self.NORMAL_CLASS_INDEX]
                    loss = -tf.math.log(target_probability + 1e-8)

                gradients = tape.gradient(loss, image_var)
                optimizer.apply_gradients([(gradients, image_var)])
                image_var.assign(tf.clip_by_value(image_var, 0.0, 1.0))

                if (step + 1) % 10 == 0 or step == steps - 1:
                    logger.info(
                        f"Counterfactual optimization step {step + 1}/{steps} — "
                        f"target-class loss: {float(tf.reduce_mean(loss)):.6f}"
                    )

            counterfactual_pixels = (image_var.numpy()[0] * 255.0).astype(np.uint8)
            counterfactual_bgr = cv2.cvtColor(counterfactual_pixels, cv2.COLOR_RGB2BGR)

            output_path = Path(self.config.counterfactual_output_dir) / f"{output_name}_counterfactual.png"
            cv2.imwrite(str(output_path), counterfactual_bgr)
            logger.info(f"Counterfactual simulation saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.exception(f"Counterfactual generation failed for '{image_path}': {e}")
            raise e
