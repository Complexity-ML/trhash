"""Framework-independent image preprocessing and geometry restoration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .metadata import ModelMetadata


@dataclass(frozen=True)
class ImageGeometry:
    width: int
    height: int
    scale: float
    left: int
    top: int


def preprocess(image: Image.Image, metadata: ModelMetadata) -> tuple[np.ndarray, ImageGeometry]:
    image = image.convert("RGB")
    width, height = image.size
    if metadata.resize_mode == "stretch":
        scale = metadata.image_size / max(width, height)
        left = top = 0
        canvas = image.resize(
            (metadata.image_size, metadata.image_size),
            Image.Resampling.BILINEAR,
        )
    else:
        scale = min(metadata.image_size / width, metadata.image_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        left = (metadata.image_size - resized_width) // 2
        top = (metadata.image_size - resized_height) // 2
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new(
            "RGB",
            (metadata.image_size, metadata.image_size),
            metadata.letterbox_color,
        )
        canvas.paste(resized, (left, top))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray(metadata.image_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(metadata.image_std, dtype=np.float32)[:, None, None]
    geometry = ImageGeometry(width, height, scale, left, top)
    return np.ascontiguousarray((pixels - mean) / std), geometry


def restore_boxes(boxes: np.ndarray, metadata: ModelMetadata, geometry: ImageGeometry) -> np.ndarray:
    restored = boxes.astype(np.float32, copy=True)
    restored[:, (0, 2)] = (
        restored[:, (0, 2)] * metadata.image_size - geometry.left
    ) / geometry.scale
    restored[:, (1, 3)] = (
        restored[:, (1, 3)] * metadata.image_size - geometry.top
    ) / geometry.scale
    restored[:, (0, 2)] = restored[:, (0, 2)].clip(0, geometry.width)
    restored[:, (1, 3)] = restored[:, (1, 3)].clip(0, geometry.height)
    return restored
