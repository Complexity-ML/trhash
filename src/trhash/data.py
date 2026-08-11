"""YOLO dataset configuration used by public fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import yaml


@dataclass(frozen=True)
class YoloDataset:
    config: Path
    train_images: Path
    train_labels: Path
    validation_images: Optional[Path]
    validation_labels: Optional[Path]
    names: Tuple[str, ...]


def _path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field}' must be a directory path")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _labels_for(images: Path) -> Path:
    parts = list(images.parts)
    positions = [index for index, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError(
            f"cannot infer labels for {images}; use images/<split> and labels/<split>"
        )
    parts[positions[-1]] = "labels"
    return Path(*parts)


def load_dataset(path: Union[str, Path]) -> YoloDataset:
    config = Path(path).expanduser().resolve()
    values = yaml.safe_load(config.read_text())
    if not isinstance(values, dict):
        raise ValueError("dataset YAML must contain a mapping")

    root = _path(config.parent, values.get("path", "."), "path")
    raw_names = values.get("names")
    if isinstance(raw_names, list):
        names = tuple(str(name) for name in raw_names)
    elif isinstance(raw_names, dict):
        try:
            indexed = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("class IDs in 'names' must be integers") from error
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("class IDs in 'names' must be contiguous from zero")
        names = tuple(indexed[index] for index in range(len(indexed)))
    else:
        raise ValueError("'names' must be a list or class-ID mapping")
    if not names:
        raise ValueError("the dataset must declare at least one class")

    train_images = _path(root, values.get("train"), "train")
    train_labels = (
        _path(root, values["train_labels"], "train_labels")
        if values.get("train_labels") is not None
        else _labels_for(train_images)
    )
    validation_images = None
    validation_labels = None
    if values.get("val") is not None:
        validation_images = _path(root, values["val"], "val")
        validation_labels = (
            _path(root, values["val_labels"], "val_labels")
            if values.get("val_labels") is not None
            else _labels_for(validation_images)
        )

    for label, directory in (
        ("training images", train_images),
        ("training labels", train_labels),
        ("validation images", validation_images),
        ("validation labels", validation_labels),
    ):
        if directory is not None and not directory.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    return YoloDataset(
        config=config,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        names=names,
    )


def observed_class_count(labels: Path) -> int:
    largest = -1
    for label_file in labels.rglob("*.txt"):
        for line in label_file.read_text().splitlines():
            if line.strip():
                largest = max(largest, int(line.split()[0]))
    return largest + 1
