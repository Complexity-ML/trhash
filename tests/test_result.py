from pathlib import Path

from PIL import Image

from trhash.result import Result


def test_result_serializes_and_renders(tmp_path: Path):
    result = Result(
        image=Image.new("RGB", (64, 32), "white"),
        boxes=[(4.0, 5.0, 30.0, 25.0)],
        scores=[0.9],
        labels=[1],
        names=("cat", "dog"),
    )

    payload = result.to_dict()
    output = result.save(tmp_path / "prediction.png")

    assert payload["detections"][0]["class_name"] == "dog"
    assert output.is_file()
