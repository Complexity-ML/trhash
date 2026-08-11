"""Class-aware IoU matching with a dependency-free Hungarian solver."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=2)
    first_area = np.prod(np.maximum(first[:, 2:] - first[:, :2], 0.0), axis=1)
    second_area = np.prod(np.maximum(second[:, 2:] - second[:, :2], 0.0), axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return the minimum-cost rectangular assignment in O(n^3)."""

    rows, columns = cost.shape
    if not rows or not columns:
        return []
    transposed = rows > columns
    matrix = cost.T if transposed else cost
    rows, columns = matrix.shape
    row_potential = np.zeros(rows + 1, dtype=np.float64)
    column_potential = np.zeros(columns + 1, dtype=np.float64)
    matched_row = np.zeros(columns + 1, dtype=np.int64)
    previous_column = np.zeros(columns + 1, dtype=np.int64)

    for row in range(1, rows + 1):
        matched_row[0] = row
        minimum = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = (
                    matrix[current_row - 1, candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = previous_column[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    pairs = [
        (int(matched_row[column] - 1), column - 1)
        for column in range(1, columns + 1)
        if matched_row[column]
    ]
    return [(column, row) for row, column in pairs] if transposed else pairs


def match_iou(
    track_boxes: np.ndarray,
    track_labels: Sequence[int],
    detection_boxes: np.ndarray,
    detection_labels: Sequence[int],
    *,
    minimum_iou: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not len(track_boxes) or not len(detection_boxes):
        return [], list(range(len(track_boxes))), list(range(len(detection_boxes)))
    similarities = box_iou(track_boxes, detection_boxes)
    same_class = np.equal.outer(np.asarray(track_labels), np.asarray(detection_labels))
    cost = np.where(same_class, 1.0 - similarities, 1e6)
    matches = [
        (track, detection)
        for track, detection in _hungarian(cost)
        if same_class[track, detection] and similarities[track, detection] >= minimum_iou
    ]
    matched_tracks = {track for track, _ in matches}
    matched_detections = {detection for _, detection in matches}
    return (
        matches,
        [index for index in range(len(track_boxes)) if index not in matched_tracks],
        [index for index in range(len(detection_boxes)) if index not in matched_detections],
    )
