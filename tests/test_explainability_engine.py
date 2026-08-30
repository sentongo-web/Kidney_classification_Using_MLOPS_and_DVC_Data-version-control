from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from cnnClassifier.components.explainability_engine import ExplainabilityEngine
from cnnClassifier.entity.config_entity import ExplainabilityConfig


def _build_mildly_biased_model() -> tf.keras.Model:
    """A model whose classification head is gently (not overwhelmingly)
    biased toward 'Tumor', so pixel-space gradient descent has realistic
    room to move the 'Normal'-class probability upward within a handful
    of optimizer steps — used only to verify the counterfactual loop's
    optimization direction, independent of the deterministic fixtures in
    conftest.py that are tuned for classification-outcome stability.
    """
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="block5_conv3")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs)

    dense_layer = model.get_layer("predictions")
    kernel, _ = dense_layer.get_weights()
    rng = np.random.default_rng(1)
    kernel = rng.normal(scale=0.05, size=kernel.shape).astype(np.float32)
    bias = np.array([-1.0, 1.0], dtype=np.float32)
    dense_layer.set_weights([kernel, bias])
    return model


def _make_config(tmp_path: Path) -> ExplainabilityConfig:
    root = tmp_path / "explainability"
    gradcam_dir = root / "gradcam"
    counterfactual_dir = root / "counterfactuals"
    gradcam_dir.mkdir(parents=True)
    counterfactual_dir.mkdir(parents=True)
    return ExplainabilityConfig(
        root_dir=root,
        gradcam_output_dir=gradcam_dir,
        counterfactual_output_dir=counterfactual_dir,
        target_layer_name="block5_conv3",
    )


def test_generate_gradcam_writes_a_readable_overlay(dummy_model, sample_image_path, tmp_path):
    config = _make_config(tmp_path)
    engine = ExplainabilityEngine(config=config)

    output_path = engine.generate_gradcam(model=dummy_model, image_path=sample_image_path, output_name="case_1")

    assert Path(output_path).is_file()
    assert Path(output_path).parent == config.gradcam_output_dir
    saved = np.array(Image.open(output_path))
    assert saved.shape[:2] == (224, 224)


def test_generate_gradcam_raises_on_missing_image(dummy_model, tmp_path):
    config = _make_config(tmp_path)
    engine = ExplainabilityEngine(config=config)

    try:
        engine.generate_gradcam(model=dummy_model, image_path=str(tmp_path / "missing.png"), output_name="case_x")
        assert False, "expected FileNotFoundError to propagate"
    except FileNotFoundError:
        pass


def test_generate_counterfactual_writes_a_readable_image(dummy_model, sample_image_path, tmp_path):
    config = _make_config(tmp_path)
    engine = ExplainabilityEngine(config=config)

    output_path = engine.generate_counterfactual(
        model=dummy_model, image_path=sample_image_path, output_name="case_2", steps=5, lr=0.05
    )

    assert Path(output_path).is_file()
    assert Path(output_path).parent == config.counterfactual_output_dir
    saved = np.array(Image.open(output_path))
    assert saved.shape == (224, 224, 3)


def test_counterfactual_optimization_moves_toward_normal_class(sample_image_path, tmp_path):
    """The gradient-descent loop should push the 'Normal'-class probability
    up relative to the original, unoptimized scan — the mathematical
    guarantee the counterfactual relies on to be meaningful clinical
    evidence ("if this tissue were healthy, pixels would look like this").
    """
    config = _make_config(tmp_path)
    engine = ExplainabilityEngine(config=config)
    model = _build_mildly_biased_model()

    baseline_array = engine._load_and_preprocess(sample_image_path, (224, 224))
    baseline_prob = float(model.predict(baseline_array, verbose=0)[0, engine.NORMAL_CLASS_INDEX])

    output_path = engine.generate_counterfactual(
        model=model, image_path=sample_image_path, output_name="case_3", steps=50, lr=0.05
    )

    optimized_array = engine._load_and_preprocess(output_path, (224, 224))
    final_prob = float(model.predict(optimized_array, verbose=0)[0, engine.NORMAL_CLASS_INDEX])

    assert final_prob > baseline_prob
