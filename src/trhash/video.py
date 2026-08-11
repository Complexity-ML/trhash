"""Lazy video capture and annotated video writing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union
from urllib.parse import urlparse

from PIL import Image

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VideoSource = Union[str, Path, int]


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('video support requires `pip install "trhash[video]"`') from error
    return cv2


def is_video_source(source) -> bool:
    if isinstance(source, int):
        return True
    if not isinstance(source, (str, Path)):
        return False
    value = str(source)
    if value.isdecimal():
        return True
    parsed = urlparse(value)
    if parsed.scheme.casefold() in {"rtmp", "rtsp", "rtsps"}:
        return True
    candidate = parsed.path if parsed.scheme else value
    return Path(candidate).suffix.casefold() in VIDEO_EXTENSIONS


def _capture_value(source: VideoSource):
    value = str(source)
    return int(value) if isinstance(source, int) or value.isdecimal() else value


@dataclass(frozen=True)
class VideoFrame:
    image: Image.Image
    source: str
    frame_index: int
    timestamp: float
    fps: float


def read_video(source: VideoSource) -> Iterator[VideoFrame]:
    """Yield RGB frames and release the capture on exhaustion or generator close."""

    cv2 = _cv2()
    capture = cv2.VideoCapture(_capture_value(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video source: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frame_index = 0
    try:
        while True:
            available, frame = capture.read()
            if not available:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else frame_index / fps
            yield VideoFrame(
                image=Image.fromarray(rgb),
                source=str(source),
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
            )
            frame_index += 1
    finally:
        capture.release()


class VideoWriter:
    def __init__(self, output: Union[str, Path], *, fps: float, size: tuple[int, int]) -> None:
        cv2 = _cv2()
        self.cv2 = cv2
        self.output = Path(output).expanduser()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        extension = self.output.suffix.casefold()
        codec = "XVID" if extension == ".avi" else "mp4v"
        self.writer = cv2.VideoWriter(
            str(self.output),
            cv2.VideoWriter_fourcc(*codec),
            fps if fps > 0 else 30.0,
            size,
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"could not create video output: {self.output}")
        self.size = size

    def write(self, image: Image.Image) -> None:
        import numpy as np

        if image.size != self.size:
            raise ValueError(f"video frame size changed from {self.size} to {image.size}")
        rgb = image.convert("RGB")
        self.writer.write(
            self.cv2.cvtColor(np.asarray(rgb), self.cv2.COLOR_RGB2BGR)
        )

    def close(self) -> Path:
        self.writer.release()
        return self.output.resolve()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_errors) -> None:
        self.close()
