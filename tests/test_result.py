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
        speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5},
        track_ids=[7],
        frame_index=12,
        timestamp=0.4,
        fps=30.0,
    )

    payload = result.to_dict()
    output = result.save(tmp_path / "prediction.png")

    assert payload["detections"][0]["class_name"] == "dog"
    assert payload["speed"]["inference"] == 2.0
    assert payload["detections"][0]["track_id"] == 7
    assert payload["frame_index"] == 12
    assert output.is_file()

    restored = Result.from_payload(result.image, payload)
    assert restored.track_ids == [7]
    assert restored.frame_index == 12


def test_result_render_options_and_show(monkeypatch):
    result = Result(
        image=Image.new("RGB", (64, 32), "white"),
        boxes=[(4.0, 5.0, 30.0, 25.0)],
        scores=[0.9],
        labels=[0],
        names=("object",),
    )
    shown = []
    monkeypatch.setattr(
        Image.Image,
        "show",
        lambda image, title=None: shown.append((image.size, title)),
    )

    rendered = result.show(labels=False, conf=False, line_width=1)

    assert rendered.size == result.image.size
    assert shown == [((64, 32), "TR-Hash prediction")]
