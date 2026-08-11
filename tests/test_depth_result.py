from pathlib import Path

import pytest
from PIL import Image

from trhash import DepthResult
from trhash.result import result_from_payload


def test_depth_result_round_trip_render_and_raw_map(tmp_path: Path):
    depth = Image.new("F", (8, 4))
    depth.putdata([float(index) / 10.0 for index in range(32)])
    result = DepthResult(
        image=Image.new("RGB", (8, 4), "white"),
        depth=depth,
        speed={"inference": 2.0},
    )

    payload = result.to_dict()
    restored = result_from_payload(result.image, payload)
    rendered = result.save(tmp_path / "depth-color.png", alpha=0.8)
    raw = result.save_depth(tmp_path / "depth.tiff")

    assert payload["depth"]["min"] == pytest.approx(0.0)
    assert payload["depth"]["max"] == pytest.approx(3.1)
    assert isinstance(restored, DepthResult)
    assert list(restored.depth.get_flattened_data()) == pytest.approx(
        list(result.depth.get_flattened_data())
    )
    assert rendered.is_file()
    assert raw.is_file()


def test_raw_depth_requires_lossless_float_format(tmp_path: Path):
    result = DepthResult(
        image=Image.new("RGB", (2, 2), "white"),
        depth=Image.new("F", (2, 2), 1.0),
    )

    with pytest.raises(ValueError, match="tif"):
        result.save_depth(tmp_path / "depth.png")
