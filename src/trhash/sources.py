"""Normalize image files, directories, and iterable prediction sources."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Union

from PIL import Image

ImageSource = Union[str, Path, Image.Image]
PredictionSource = Union[ImageSource, Iterable[ImageSource]]
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


def expand_sources(source: PredictionSource) -> tuple[Iterator[ImageSource], bool]:
    """Return a source iterator and whether the request is exactly one image."""

    if isinstance(source, Image.Image):
        return iter((source,)), True
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_dir():
            return iter(image_files(path)), False
        return iter((source,)), True
    if not isinstance(source, Iterable):
        raise TypeError("source must be an image, path, directory, or iterable of images")
    return iter(source), False


def chunks(sources: Iterator[ImageSource], size: int) -> Iterator[list[ImageSource]]:
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
