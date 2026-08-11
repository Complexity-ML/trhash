from pathlib import Path

from PIL import Image

from trhash import ClassificationResult
from trhash.result import result_from_payload


def test_classification_result_round_trip_and_render(tmp_path: Path):
    result = ClassificationResult(
        image=Image.new("RGB", (64, 32), "white"),
        scores=[0.8, 0.15, 0.05],
        labels=[1, 0, 2],
        names=("cat", "dog", "bird"),
        speed={"inference": 2.0},
    )

    payload = result.to_dict(top_k=2)
    restored = result_from_payload(result.image, payload)
    output = result.save(tmp_path / "classification.png", top_k=2)

    assert payload["predictions"][0]["class_name"] == "dog"
    assert isinstance(restored, ClassificationResult)
    assert restored.top1 == 1
    assert output.is_file()
