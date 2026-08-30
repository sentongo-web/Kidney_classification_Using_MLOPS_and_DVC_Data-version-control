import io

import numpy as np
import pytest
from PIL import Image

import app as app_module
from cnnClassifier.components.explainability_engine import ExplainabilityEngine
from cnnClassifier.entity.config_entity import ExplainabilityConfig


def _make_upload_bytes() -> bytes:
    rng = np.random.default_rng(7)
    array = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


def _make_xai_config(tmp_path) -> ExplainabilityConfig:
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


@pytest.fixture
def client(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(app_module, "_MODEL_ERROR", None)
    monkeypatch.setattr(app_module, "UPLOAD_FOLDER", str(upload_dir))
    app_module.app.testing = True
    return app_module.app.test_client()


def test_home_route_serves_ui(client):
    response = client.get("/")
    assert response.status_code == 200


def test_predict_route_rejects_missing_file(client, monkeypatch, dummy_normal_model):
    monkeypatch.setattr(app_module, "_MODEL", dummy_normal_model)
    response = client.post("/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_predict_route_returns_503_when_model_unavailable(client, monkeypatch):
    monkeypatch.setattr(app_module, "_MODEL", None)
    monkeypatch.setattr(app_module, "_MODEL_ERROR", "model file not found")
    data = {"file": (io.BytesIO(_make_upload_bytes()), "scan.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 503


def test_predict_route_returns_normal_with_no_xai_payload(client, monkeypatch, dummy_normal_model):
    monkeypatch.setattr(app_module, "_MODEL", dummy_normal_model)
    data = {"file": (io.BytesIO(_make_upload_bytes()), "scan.png")}

    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload[0]["image"] == "Normal"
    assert payload[0]["gradcam"] == ""
    assert payload[0]["counterfactual"] == ""
    assert "gradcam_path" not in payload[0]
    assert "counterfactual_path" not in payload[0]


def test_predict_route_returns_tumor_with_base64_xai_payload(client, monkeypatch, dummy_tumor_model, tmp_path):
    monkeypatch.setattr(app_module, "_MODEL", dummy_tumor_model)
    monkeypatch.setattr(
        app_module, "_XAI_ENGINE", ExplainabilityEngine(config=_make_xai_config(tmp_path))
    )
    data = {"file": (io.BytesIO(_make_upload_bytes()), "scan.png")}

    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload[0]["image"] == "Tumor"
    assert payload[0]["gradcam"] != ""
    assert payload[0]["counterfactual"] != ""


def test_predict_route_survives_xai_engine_failure(client, monkeypatch, dummy_tumor_model):
    """Even if the explainability engine is completely broken, uploading
    an image must still return a classification instead of a 500 error.
    """
    from unittest.mock import MagicMock

    broken_engine = MagicMock()
    broken_engine.generate_gradcam.side_effect = RuntimeError("engine unavailable")

    monkeypatch.setattr(app_module, "_MODEL", dummy_tumor_model)
    monkeypatch.setattr(app_module, "_XAI_ENGINE", broken_engine)
    data = {"file": (io.BytesIO(_make_upload_bytes()), "scan.png")}

    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload[0]["image"] == "Tumor"
    assert payload[0]["gradcam"] == ""
    assert payload[0]["counterfactual"] == ""


def test_health_route_reports_model_status(client, monkeypatch, dummy_normal_model):
    monkeypatch.setattr(app_module, "_MODEL", dummy_normal_model)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True
