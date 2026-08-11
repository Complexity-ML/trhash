"""ByteTrack-style two-stage association for TR-Hash detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .matching import match_iou
from .motion import BoxKalman


@dataclass
class _Track:
    track_id: int
    label: int
    score: float
    motion: BoxKalman
    missed: int = 0
    hits: int = 1

    @property
    def box(self) -> np.ndarray:
        return self.motion.box

    def predict(self) -> None:
        self.motion.predict()
        self.missed += 1

    def update(self, box, score: float) -> None:
        self.motion.update(box)
        self.score = float(score)
        self.missed = 0
        self.hits += 1


class ByteTracker:
    """Track high-confidence boxes, recovering tracks with low-score boxes."""

    def __init__(
        self,
        *,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        new_track_threshold: Optional[float] = None,
        match_iou_threshold: float = 0.3,
        second_match_iou_threshold: float = 0.2,
        track_buffer: int = 30,
    ) -> None:
        if not 0.0 <= low_threshold <= high_threshold <= 1.0:
            raise ValueError("tracking thresholds must satisfy 0 <= low <= high <= 1")
        if not 0.0 <= match_iou_threshold <= 1.0:
            raise ValueError("match_iou_threshold must be between 0 and 1")
        if not 0.0 <= second_match_iou_threshold <= 1.0:
            raise ValueError("second_match_iou_threshold must be between 0 and 1")
        if track_buffer < 1:
            raise ValueError("track_buffer must be positive")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.new_track_threshold = (
            high_threshold if new_track_threshold is None else new_track_threshold
        )
        if not 0.0 <= self.new_track_threshold <= 1.0:
            raise ValueError("new_track_threshold must be between 0 and 1")
        self.match_iou_threshold = match_iou_threshold
        self.second_match_iou_threshold = second_match_iou_threshold
        self.track_buffer = track_buffer
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_Track] = []
        self._next_id = 1

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(track.track_id for track in self._tracks if track.missed == 0)

    def _associate(
        self,
        track_indices: Sequence[int],
        detection_indices: Sequence[int],
        boxes: np.ndarray,
        labels: Sequence[int],
        minimum_iou: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        matches, unmatched_tracks, unmatched_detections = match_iou(
            np.asarray([self._tracks[index].box for index in track_indices], dtype=np.float32),
            [self._tracks[index].label for index in track_indices],
            boxes[list(detection_indices)],
            [labels[index] for index in detection_indices],
            minimum_iou=minimum_iou,
        )
        return (
            [(track_indices[track], detection_indices[detection]) for track, detection in matches],
            [track_indices[index] for index in unmatched_tracks],
            [detection_indices[index] for index in unmatched_detections],
        )

    def update(
        self,
        boxes: Sequence[Sequence[float]],
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> list[Optional[int]]:
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError("boxes, scores, and labels must have equal lengths")
        detections = np.asarray(boxes, dtype=np.float32).reshape((-1, 4))
        scores = [float(score) for score in scores]
        labels = [int(label) for label in labels]
        assignments: list[Optional[int]] = [None] * len(detections)

        for track in self._tracks:
            track.predict()

        high = [index for index, score in enumerate(scores) if score >= self.high_threshold]
        low = [
            index
            for index, score in enumerate(scores)
            if self.low_threshold <= score < self.high_threshold
        ]
        track_indices = list(range(len(self._tracks)))
        matches, unmatched_tracks, unmatched_high = self._associate(
            track_indices,
            high,
            detections,
            labels,
            self.match_iou_threshold,
        )
        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            track.update(detections[detection_index], scores[detection_index])
            assignments[detection_index] = track.track_id

        recently_lost = [
            index for index in unmatched_tracks if self._tracks[index].missed == 1
        ]
        second_matches, _, _ = self._associate(
            recently_lost,
            low,
            detections,
            labels,
            self.second_match_iou_threshold,
        )
        for track_index, detection_index in second_matches:
            track = self._tracks[track_index]
            track.update(detections[detection_index], scores[detection_index])
            assignments[detection_index] = track.track_id

        for detection_index in unmatched_high:
            if scores[detection_index] < self.new_track_threshold:
                continue
            track = _Track(
                track_id=self._next_id,
                label=labels[detection_index],
                score=scores[detection_index],
                motion=BoxKalman(detections[detection_index]),
            )
            self._next_id += 1
            self._tracks.append(track)
            assignments[detection_index] = track.track_id

        self._tracks = [track for track in self._tracks if track.missed <= self.track_buffer]
        return assignments
