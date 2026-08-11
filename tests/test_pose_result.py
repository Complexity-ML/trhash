from pathlib import Path

import pytest
from PIL import Image

from trhash import PoseResult
from trhash.result import result_from_payload


def test_pose_result_transport_and_rendering(tmp_path: Path):
    image = Image.new("RGB", (64, 32), "white")
    result = PoseResult(
        image=image,
        keypoints=[(10.0, 10.0, 0.9), (40.0, 20.0, 0.8)],
        names=("nose", "tail"),
        source="animal.jpg",
        speed={"inference": 2.5},
    )

    payload = result.to_dict()
    restored = result_from_payload(image, payload)
    output = restored.save(
        tmp_path / "pose.jpg",
        connections=((0, 1),),
        conf=True,
    )

    assert isinstance(restored, PoseResult)
    assert restored.names == ("nose", "tail")
    assert restored.keypoints == pytest.approx(result.keypoints)
    assert restored.source == "animal.jpg"
    assert restored.speed == {"inference": 2.5}
    assert output.is_file()


def test_pose_result_rejects_misaligned_names():
    with pytest.raises(ValueError, match="equal lengths"):
        PoseResult(
            image=Image.new("RGB", (16, 16)),
            keypoints=[(1.0, 2.0, 0.9)],
            names=(),
        )
