from types import SimpleNamespace

import pytest
from PIL import Image

from trhash.result import Result

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


class FakeBackend:
    def __init__(self, model, device=None):
        self.model_id = str(model)
        self.providers = [device or "CPUExecutionProvider"]
        self.metadata = SimpleNamespace(num_classes=1, format_version=2, task="detection")

    def predict(self, image, **_options):
        return Result(image, [(1, 2, 10, 12)], [0.9], [0], ("object",))


def test_server_health_authentication_and_prediction(monkeypatch, tmp_path):
    import trhash.server.app as server_app

    monkeypatch.setattr(server_app, "load_portable_backend", FakeBackend)
    app = server_app.create_app("model", api_key="secret")
    client = TestClient(app)
    image = tmp_path / "image.jpg"
    Image.new("RGB", (20, 20), "white").save(image)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").json() == {
        "ready": True,
        "task": "detection",
        "num_classes": 1,
    }
    assert client.get("/v1/model").status_code == 401
    with image.open("rb") as handle:
        response = client.post(
            "/v1/predict",
            headers={"X-API-Key": "secret"},
            files={"file": ("image.jpg", handle, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["detections"][0]["class_name"] == "object"
