from pathlib import Path

from PIL import Image, ImageDraw

from trhash import InstanceSegmentationResult
from trhash.result import result_from_payload


def test_instance_result_transport_and_rendering(tmp_path: Path):
    image = Image.new("RGB", (64, 32), "white")
    mask = Image.new("L", image.size)
    ImageDraw.Draw(mask).rectangle((8, 6, 40, 26), fill=255)
    result = InstanceSegmentationResult(
        image=image,
        boxes=[(8.0, 6.0, 40.0, 26.0)],
        masks=[mask],
        scores=[0.91],
        labels=[1],
        names=("cat", "dog"),
        source="animal.jpg",
        speed={"inference": 3.0},
    )

    payload = result.to_dict()
    restored = result_from_payload(image, payload)
    output = restored.save(tmp_path / "instances.jpg")

    assert isinstance(restored, InstanceSegmentationResult)
    assert restored.boxes == result.boxes
    assert restored.scores == result.scores
    assert restored.names == ("0", "dog")
    assert restored.masks[0].tobytes() == mask.tobytes()
    assert restored.source == "animal.jpg"
    assert output.is_file()
