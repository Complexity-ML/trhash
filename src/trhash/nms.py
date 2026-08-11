"""Small NumPy fallback NMS for legacy one-to-many exports."""

from __future__ import annotations

import numpy as np


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        top_left = np.maximum(boxes[current, :2], boxes[remaining, :2])
        bottom_right = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        intersection = np.maximum(bottom_right - top_left, 0).prod(axis=1)
        area_a = np.maximum(boxes[current, 2:] - boxes[current, :2], 0).prod()
        area_b = np.maximum(boxes[remaining, 2:] - boxes[remaining, :2], 0).prod(axis=1)
        iou = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
        order = remaining[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    max_detections: int,
) -> np.ndarray:
    kept = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        kept.extend(indices[_nms(boxes[indices], scores[indices], threshold)].tolist())
    return np.asarray(sorted(kept, key=lambda index: scores[index], reverse=True)[:max_detections])
