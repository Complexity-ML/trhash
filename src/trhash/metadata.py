"""Portable model-bundle metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union


@dataclass(frozen=True)
class ModelMetadata:
    format_version: int
    task: str
    model_file: str
    image_size: int
    num_classes: int
    class_names: Tuple[str, ...]
    grid_sizes: Tuple[int, ...]
    reg_max: int
    box_encoding: str
    score_encoding: str
    recommended_confidence: float
    letterbox_color: Tuple[int, int, int] = (114, 114, 114)
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelMetadata":
        values = dict(values)
        for field in ("class_names", "grid_sizes", "letterbox_color", "image_mean", "image_std"):
            values[field] = tuple(values[field])
        metadata = cls(**values)
        if metadata.format_version != 2 or metadata.task != "detection":
            raise ValueError("unsupported TR-Hash model bundle")
        if Path(metadata.model_file).name != metadata.model_file:
            raise ValueError("model_file must be a filename inside the bundle")
        if len(metadata.class_names) != metadata.num_classes:
            raise ValueError("class_names must match num_classes")
        if metadata.reg_max < 0 or metadata.reg_max == 1:
            raise ValueError("reg_max must be 0 or at least 2")
        if metadata.box_encoding != "stride_ltrb_dfl":
            raise ValueError("unsupported box encoding")
        if metadata.score_encoding != "quality_class_sigmoid":
            raise ValueError("unsupported score encoding")
        return metadata

    @classmethod
    def load(cls, bundle: Union[str, Path]) -> "ModelMetadata":
        return cls.from_dict(json.loads((Path(bundle) / "trhash.json").read_text()))

    def save(self, bundle: Union[str, Path]) -> Path:
        path = Path(bundle) / "trhash.json"
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


def metadata_from_checkpoint(backend, model_file: str = "model.onnx") -> ModelMetadata:
    config = backend.model.config
    return ModelMetadata(
        format_version=2,
        task="detection",
        model_file=model_file,
        image_size=config.image_size,
        num_classes=config.num_classes,
        class_names=tuple(backend.names),
        grid_sizes=tuple(config.grid_sizes),
        reg_max=config.reg_max,
        box_encoding="stride_ltrb_dfl",
        score_encoding="quality_class_sigmoid",
        recommended_confidence=float(backend.validation.get("best_confidence", 0.25)),
    )
