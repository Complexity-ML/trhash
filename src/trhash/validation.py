"""Framework-independent Pascal-style detection validation at IoU 0.50."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from .data import load_dataset
from .result import Result
from .sources import image_files

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class ValidationMetrics:
    map50: float
    precision: float
    recall: float
    f1: float
    best_confidence: float
    images: int
    targets: int
    predictions: int
    per_class_ap50: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _iou(first: Box, second: Box) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(second[3] - second[1], 0.0)
    return intersection / max(first_area + second_area - intersection, 1e-9)


def _targets(path: Path, width: int, height: int, num_classes: int) -> list[tuple[Box, int]]:
    if not path.is_file():
        return []
    targets = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) < 5:
            raise ValueError(f"invalid YOLO row at {path}:{line_number}")
        class_id = int(values[0])
        if not 0 <= class_id < num_classes:
            raise ValueError(f"class ID out of range at {path}:{line_number}")
        center_x, center_y, box_width, box_height = map(float, values[1:5])
        center_x *= width
        center_y *= height
        box_width *= width
        box_height *= height
        targets.append(
            (
                (
                    center_x - box_width / 2,
                    center_y - box_height / 2,
                    center_x + box_width / 2,
                    center_y + box_height / 2,
                ),
                class_id,
            )
        )
    return targets


def _match(
    result: Result,
    targets: list[tuple[Box, int]],
    records: dict[int, list[tuple[float, bool]]],
    *,
    match_iou: float,
) -> None:
    for class_id in range(len(result.names)):
        class_targets = [box for box, label in targets if label == class_id]
        used = [False] * len(class_targets)
        predictions = sorted(
            (
                (score, box)
                for box, score, label in zip(result.boxes, result.scores, result.labels)
                if label == class_id
            ),
            reverse=True,
        )
        for score, box in predictions:
            candidates = [
                (_iou(box, target), index)
                for index, target in enumerate(class_targets)
                if not used[index]
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            matched = best_iou >= match_iou
            if matched:
                used[best_index] = True
            records[class_id].append((float(score), matched))


def _average_precision(records: list[tuple[float, bool]], target_count: int) -> float:
    if target_count == 0:
        return 0.0
    ordered = sorted(records, reverse=True)
    true_positives = 0
    points = []
    for index, (_, matched) in enumerate(ordered, start=1):
        true_positives += int(matched)
        points.append((true_positives / target_count, true_positives / index))
    return sum(
        max((precision for recall, precision in points if recall >= threshold), default=0.0)
        for threshold in (index / 100 for index in range(101))
    ) / 101


def _best_operating_point(
    records: Iterable[tuple[float, bool]],
    target_count: int,
) -> tuple[float, float, float, float]:
    ordered = sorted(records, reverse=True)
    best = (0.0, 0.0, 0.0, 1.0)
    true_positives = 0
    for index, (score, matched) in enumerate(ordered, start=1):
        true_positives += int(matched)
        precision = true_positives / index
        recall = true_positives / max(target_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best[2]:
            best = (precision, recall, f1, score)
    return best


def validate(
    model,
    *,
    data: Union[str, Path],
    confidence: float = 0.001,
    iou: float = 0.45,
    match_iou: float = 0.50,
    batch: int = 16,
    max_images: Optional[int] = None,
) -> ValidationMetrics:
    dataset = load_dataset(data)
    if dataset.validation_images is None or dataset.validation_labels is None:
        raise ValueError("dataset YAML must define a validation split with 'val'")
    backend_names = tuple(str(name) for name in getattr(model.backend, "names", ()))
    if backend_names and tuple(name.casefold() for name in backend_names) != tuple(
        name.casefold() for name in dataset.names
    ):
        raise ValueError("model class names do not match dataset names")
    images = image_files(dataset.validation_images)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        images = images[:max_images]

    records = {class_id: [] for class_id in range(len(dataset.names))}
    target_counts = [0] * len(dataset.names)
    prediction_count = 0
    results = model.predict(
        images,
        confidence=confidence,
        iou=iou,
        batch=batch,
        stream=True,
    )
    for image_path, result in zip(images, results):
        relative = image_path.relative_to(dataset.validation_images).with_suffix(".txt")
        targets = _targets(
            dataset.validation_labels / relative,
            result.image.width,
            result.image.height,
            len(dataset.names),
        )
        for _, class_id in targets:
            target_counts[class_id] += 1
        prediction_count += len(result.boxes)
        _match(result, targets, records, match_iou=match_iou)

    per_class = {
        dataset.names[class_id]: _average_precision(records[class_id], target_counts[class_id])
        for class_id in range(len(dataset.names))
        if target_counts[class_id]
    }
    all_records = [record for class_records in records.values() for record in class_records]
    precision, recall, f1, best_confidence = _best_operating_point(
        all_records,
        sum(target_counts),
    )
    return ValidationMetrics(
        map50=sum(per_class.values()) / max(len(per_class), 1),
        precision=precision,
        recall=recall,
        f1=f1,
        best_confidence=best_confidence,
        images=len(images),
        targets=sum(target_counts),
        predictions=prediction_count,
        per_class_ap50=per_class,
    )
