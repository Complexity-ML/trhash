"""Normalize image files, directories, and iterable prediction sources."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Union

from PIL import Image

ImageSource = Union[str, Path, Image.Image]
PredictionSource = Union[ImageSource, int, Iterable[ImageSource]]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def image_files(directory: Path) -> list[Path]:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"no supported images found in {directory}")
    return files


def expand_sources(source: PredictionSource) -> tuple[Iterator, bool]:
    """Return a source iterator and whether the request is exactly one image."""

    if isinstance(source, Image.Image):
        return iter((source,)), True
    from .video import is_video_source, read_video

    if is_video_source(source):
        return read_video(source), False
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_dir():
            return iter(image_files(path)), False
        return iter((source,)), True
    if not isinstance(source, Iterable):
        raise TypeError("source must be an image, image directory, video, stream, or iterable")
    return iter(source), False


def inference_source(source):
    from .video import VideoFrame

    return source.image if isinstance(source, VideoFrame) else source


def attach_source_metadata(result, source) -> None:
    from .video import VideoFrame

    if not isinstance(source, VideoFrame):
        return
    result.source = source.source
    result.frame_index = source.frame_index
    result.timestamp = source.timestamp
    result.fps = source.fps


def chunks(sources: Iterator, size: int) -> Iterator[list]:
    if size <= 0:
        raise ValueError("batch must be positive")
    while True:
        batch = []
        for _ in range(size):
            try:
                batch.append(next(sources))
            except StopIteration:
                break
        if not batch:
            return
        yield batch
