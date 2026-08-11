"""Decode portable detector outputs using only NumPy."""

from __future__ import annotations

import numpy as np

from .metadata import ModelMetadata
from .nms import class_aware_nms


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0)


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
    return_indices: bool = False,
):
    bins = metadata.reg_max + 1 if metadata.reg_max else 1
    regression_width = 4 * bins
    if raw.ndim != 2 or raw.shape[1] != regression_width + metadata.num_classes:
        raise ValueError(f"unexpected detector output shape: {raw.shape}")
    rows, columns, denominators = _geometry(metadata.grid_sizes)
    if raw.shape[0] != rows.size:
        raise ValueError("detector cells do not match bundle grid_sizes")
    center_x = (columns + 0.5) / denominators
    center_y = (rows + 0.5) / denominators
    regression = raw[:, :regression_width]
    if metadata.reg_max:
        distributions = _softmax(regression.reshape(-1, 4, bins))
        distances = (distributions * np.arange(bins, dtype=np.float32)).sum(axis=-1)
    else:
        distances = _softplus(regression)
    distances /= denominators[:, None]
    left, top, right, bottom = distances.T
    boxes = np.stack(
        (
            center_x - left,
            center_y - top,
            center_x + right,
            center_y + bottom,
        ),
        axis=1,
    ).clip(0, 1)
    class_scores = _sigmoid(raw[:, regression_width:])
    labels = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(labels)), labels]
    selected = np.flatnonzero(scores >= confidence)
    boxes, scores, labels = boxes[selected], scores[selected], labels[selected]
    keep = class_aware_nms(boxes, scores, labels, iou, max_detections)
    result = boxes[keep], scores[keep], labels[keep]
    if return_indices:
        return (*result, selected[keep])
    return result
