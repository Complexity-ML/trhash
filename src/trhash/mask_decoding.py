"""NumPy/Pillow post-processing for prototype instance masks."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .preprocessing import ImageGeometry


def decode_instance_masks(
    coefficients: np.ndarray,
    prototypes: np.ndarray,
    cell_indices: np.ndarray,
    normalized_boxes: np.ndarray,
    geometry: ImageGeometry,
    *,
    image_size: int,
    threshold: float,
) -> list[Image.Image]:
    if len(cell_indices) == 0:
        return []
    selected = coefficients[cell_indices]
    logits = np.einsum("np,phw->nhw", selected, prototypes)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    resized_width = max(1, round(geometry.width * geometry.scale))
    resized_height = max(1, round(geometry.height * geometry.scale))
    content_box = (
        geometry.left,
        geometry.top,
        geometry.left + resized_width,
        geometry.top + resized_height,
    )
    masks = []
    for probability, box in zip(probabilities, normalized_boxes):
        canvas = Image.fromarray(probability.astype(np.float32), mode="F").resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )
        values = np.asarray(canvas, dtype=np.float32).copy()
        x1, y1, x2, y2 = box * image_size
        left = max(0, min(image_size, int(np.floor(x1))))
        top = max(0, min(image_size, int(np.floor(y1))))
        right = max(0, min(image_size, int(np.ceil(x2))))
        bottom = max(0, min(image_size, int(np.ceil(y2))))
        cropped = np.zeros_like(values)
        cropped[top:bottom, left:right] = values[top:bottom, left:right]
        restored = Image.fromarray(cropped, mode="F").crop(content_box).resize(
            (geometry.width, geometry.height),
            Image.Resampling.BILINEAR,
        )
        binary = (np.asarray(restored, dtype=np.float32) >= threshold).astype(np.uint8)
        masks.append(Image.fromarray(binary * 255, mode="L"))
    return masks
