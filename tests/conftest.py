from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

TARGET_LAYER_NAME = "block5_conv3"


def _build_dummy_model(bias: Tuple[float, float]) -> tf.keras.Model:
    """Builds a tiny convolutional network standing in for the trained
    VGG16 classifier during tests.

    It exposes a layer literally named 'block5_conv3' so it is compatible
    with the real ExplainabilityConfig.target_layer_name without any
    test-only overrides, keeping the Grad-CAM / counterfactual code paths
    exercised exactly as they run in production.

    The final Dense layer's kernel is initialised to small random weights
    (so real, non-degenerate gradients still flow back to the input pixels
    for Grad-CAM) while its bias is set large enough to dominate the logits,
    making the predicted class deterministic across arbitrary input images.
    """
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name=TARGET_LAYER_NAME)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="dummy_vgg_stub")

    dense_layer = model.get_layer("predictions")
    kernel, _ = dense_layer.get_weights()
    rng = np.random.default_rng(0)
    small_kernel = rng.normal(scale=0.01, size=kernel.shape).astype(np.float32)
    dense_layer.set_weights([small_kernel, np.array(bias, dtype=np.float32)])
    return model


@pytest.fixture
def dummy_model() -> tf.keras.Model:
    """A neutral dummy model with no strong class bias."""
    return _build_dummy_model(bias=(0.0, 0.0))


@pytest.fixture
def dummy_tumor_model() -> tf.keras.Model:
    """A dummy model whose bias deterministically favors class index 1 (Tumor)."""
    return _build_dummy_model(bias=(-6.0, 6.0))


@pytest.fixture
def dummy_normal_model() -> tf.keras.Model:
    """A dummy model whose bias deterministically favors class index 0 (Normal)."""
    return _build_dummy_model(bias=(6.0, -6.0))


@pytest.fixture
def sample_image_path(tmp_path) -> str:
    """Writes a small synthetic RGB 'CT scan' to disk, standing in for a
    real upload in tests that only need a readable image file.
    """
    rng = np.random.default_rng(42)
    array = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    image_path = tmp_path / "sample_scan.png"
    Image.fromarray(array).save(image_path)
    return str(image_path)
