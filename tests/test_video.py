from pathlib import Path

import pytest
from PIL import Image

from trhash import Result, Vision
from trhash.video import VideoWriter, is_video_source


class VideoBackend:
    def predict_batch(self, sources, **_options):
        return [
            Result(
                image=image.copy(),
                boxes=[],
                scores=[],
                labels=[],
                names=("object",),
            )
            for image in sources
        ]


def _video(path: Path, frames: int = 3) -> None:
    pytest.importorskip("cv2")
    with VideoWriter(path, fps=12.0, size=(32, 24)) as writer:
        for index in range(frames):
            writer.write(Image.new("RGB", (32, 24), (index * 30, 0, 0)))


def test_video_prediction_preserves_frame_metadata(tmp_path: Path):
    source = tmp_path / "input.mp4"
    _video(source)
    model = Vision.__new__(Vision)
    model.backend = VideoBackend()

    results = list(model.predict(source, batch=2, stream=True))

    assert [result.frame_index for result in results] == [0, 1, 2]
    assert all(result.source == str(source) for result in results)
    assert all(result.fps == pytest.approx(12.0, rel=0.1) for result in results)


def test_video_writer_round_trip_and_source_detection(tmp_path: Path):
    source = tmp_path / "input.mp4"
    _video(source, frames=2)

    assert is_video_source(source)
    assert is_video_source("rtsp://camera/live")
    assert is_video_source(0)
    assert source.stat().st_size > 0
