from pathlib import Path

from PIL import Image

from trhash import SemanticSegmentationResult
from trhash.result import result_from_payload


def test_semantic_result_round_trip_render_and_raw_mask(tmp_path: Path):
    mask = Image.new("I", (8, 4), 0)
    for x in range(4, 8):
        for y in range(4):
            mask.putpixel((x, y), 1)
    result = SemanticSegmentationResult(
        image=Image.new("RGB", (8, 4), "white"),
        mask=mask,
        names=("background", "object"),
        speed={"inference": 2.0},
    )

    payload = result.to_dict()
    restored = result_from_payload(result.image, payload)
    rendered = result.save(tmp_path / "overlay.png", alpha=0.5)
    raw = result.save_mask(tmp_path / "mask.png")

    assert payload["segments"][0]["fraction"] == 0.5
    assert isinstance(restored, SemanticSegmentationResult)
    assert restored.labels == (0, 1)
    assert list(restored.mask.get_flattened_data()) == list(
        result.mask.get_flattened_data()
    )
    assert rendered.is_file()
    assert raw.is_file()
