import httpx
from PIL import Image

from trhash import Vision


def test_remote_backend_needs_no_local_runtime(tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (40, 20), "white").save(source)
    vision = Vision("org/model", endpoint="https://vision.test")

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/predict"
        return httpx.Response(
            200,
            json={
                "detections": [
                    {
                        "box_xyxy": [1, 2, 20, 10],
                        "score": 0.8,
                        "label": 3,
                        "class_name": "dog",
                    }
                ]
            },
        )

    vision.backend.client.close()
    vision.backend.client = httpx.Client(transport=httpx.MockTransport(respond))

    result = vision(source)

    assert result.labels == [3]
    assert result.to_dict()["detections"][0]["class_name"] == "dog"
    vision.close()
