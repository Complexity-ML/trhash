"""Decode portable detector outputs using only NumPy."""

from __future__ import annotations

import numpy as np

from .metadata import ModelMetadata
from .nms import class_aware_nms


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _geometry(grid_sizes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns, denominators = [], [], []
    for grid in grid_sizes:
        rows.append(np.repeat(np.arange(grid), grid))
        columns.append(np.tile(np.arange(grid), grid))
        denominators.append(np.full(grid * grid, grid))
    return tuple(np.concatenate(values).astype(np.float32) for values in (rows, columns, denominators))


def decode(
    raw: np.ndarray,
    metadata: ModelMetadata,
    *,
    confidence: float,
    iou: float,
    max_detections: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if raw.ndim != 2 or raw.shape[1] != 5 + metadata.num_classes:
        raise ValueError(f"unexpected detector output shape: {raw.shape}")
    rows, columns, denominators = _geometry(metadata.grid_sizes)
    if raw.shape[0] != rows.size:
        raise ValueError("detector cells do not match bundle grid_sizes")
    center_x = raw[:, 0]
    center_y = raw[:, 1]
    if metadata.center_offset_mode == "linear":
        center_x = (columns + 0.5 + center_x) / denominators
        center_y = (rows + 0.5 + center_y) / denominators
    else:
        center_x = (columns + _sigmoid(center_x)) / denominators
        center_y = (rows + _sigmoid(center_y)) / denominators
    width, height = _sigmoid(raw[:, 2]), _sigmoid(raw[:, 3])
    boxes = np.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        axis=1,
    ).clip(0, 1)
    objectness = _sigmoid(raw[:, 4])
    logits = raw[:, 5:] - raw[:, 5:].max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = probabilities.argmax(axis=1)
    scores = objectness * probabilities[np.arange(len(labels)), labels]
    selected = np.flatnonzero(scores >= confidence)
    boxes, scores, labels = boxes[selected], scores[selected], labels[selected]
    if metadata.end_to_end:
        keep = scores.argsort()[::-1][:max_detections]
    else:
        keep = class_aware_nms(boxes, scores, labels, iou, max_detections)
    return boxes[keep], scores[keep], labels[keep]
