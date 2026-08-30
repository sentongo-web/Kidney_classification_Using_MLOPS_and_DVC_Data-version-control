from pathlib import Path

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import ExplainabilityConfig


def test_get_explainability_config_returns_typed_entity_and_creates_directories():
    manager = ConfigurationManager()

    config = manager.get_explainability_config()

    assert isinstance(config, ExplainabilityConfig)
    assert config.target_layer_name == "block5_conv3"
    assert Path(config.root_dir).is_dir()
    assert Path(config.gradcam_output_dir).is_dir()
    assert Path(config.counterfactual_output_dir).is_dir()


def test_explainability_config_paths_are_nested_under_root():
    manager = ConfigurationManager()

    config = manager.get_explainability_config()

    assert str(config.gradcam_output_dir).startswith(str(config.root_dir))
    assert str(config.counterfactual_output_dir).startswith(str(config.root_dir))
