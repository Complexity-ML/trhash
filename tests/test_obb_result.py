from pathlib import Path

import pytest
from PIL import Image

from trhash import OBBResult
from trhash.result import result_from_payload


def test_obb_result_transport_and_rendering(tmp_path: Path):
    image = Image.new("RGB", (64, 32), "white")
    result = OBBResult(
        image=image,
        boxes=[(8.0, 6.0, 40.0, 26.0, 0.4)],
        scores=[0.91],
        labels=[1],
        names=("car", "plane"),
        source="aerial.jpg",
        speed={"inference": 1.7},
    )

    payload = result.to_dict()
    restored = result_from_payload(image, payload)
    output = restored.save(tmp_path / "obb.jpg")

    assert isinstance(restored, OBBResult)
    assert restored.boxes == pytest.approx(result.boxes)
    assert restored.scores == result.scores
    assert restored.names == ("0", "plane")
    assert restored.source == "aerial.jpg"
    assert output.is_file()
