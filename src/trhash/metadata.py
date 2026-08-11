"""Portable model-bundle metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Union

SUPPORTED_TASKS = {
    "detection",
    "instance_segmentation",
    "semantic_segmentation",
    "depth",
    "classification",
    "pose",
    "obb",
}


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
    output_names: Tuple[str, ...] = ("predictions",)
    resize_mode: str = "letterbox"
    task_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelMetadata":
        values = dict(values)
        values.setdefault("output_names", ("predictions",))
        values.setdefault("resize_mode", "letterbox")
        values.setdefault("task_options", {})
        for field_name in (
            "class_names",
            "grid_sizes",
            "letterbox_color",
            "image_mean",
            "image_std",
            "output_names",
        ):
            values[field_name] = tuple(values[field_name])
        metadata = cls(**values)
        if metadata.format_version != 4 or metadata.task not in SUPPORTED_TASKS:
            raise ValueError("unsupported TR-Hash model bundle")
        if Path(metadata.model_file).name != metadata.model_file:
            raise ValueError("model_file must be a filename inside the bundle")
        if len(metadata.class_names) != metadata.num_classes:
            raise ValueError("class_names must match num_classes")
        if not metadata.output_names or len(set(metadata.output_names)) != len(
            metadata.output_names
        ):
            raise ValueError("output_names must be non-empty and unique")
        if metadata.resize_mode not in {"letterbox", "stretch"}:
            raise ValueError("unsupported resize mode")
        if not isinstance(metadata.task_options, dict):
            raise ValueError("task_options must be an object")
        if metadata.task in {"detection", "instance_segmentation", "obb"}:
            if metadata.reg_max < 0 or metadata.reg_max == 1:
                raise ValueError("reg_max must be 0 or at least 2")
            if metadata.box_encoding != "stride_ltrb_dfl":
                raise ValueError("unsupported box encoding")
            if metadata.score_encoding != "quality_class_sigmoid":
                raise ValueError("unsupported score encoding")
            if metadata.task == "detection" and metadata.output_names != (
                "predictions",
            ):
                raise ValueError("detection bundles require a predictions output")
            if metadata.task == "instance_segmentation":
                expected_outputs = (
                    "predictions",
                    "mask_coefficients",
                    "prototypes",
                )
                if metadata.output_names != expected_outputs:
                    raise ValueError("instance bundles require mask graph outputs")
                if int(metadata.task_options.get("num_prototypes", 0)) <= 0:
                    raise ValueError("instance bundles require num_prototypes")
        elif metadata.task == "classification":
            if metadata.score_encoding != "softmax":
                raise ValueError("classification bundles require softmax scores")
            if metadata.output_names != ("logits",):
                raise ValueError("classification bundles require a logits output")
        elif metadata.task == "semantic_segmentation":
            if metadata.score_encoding != "per_pixel_softmax":
                raise ValueError("semantic bundles require per-pixel softmax scores")
            if metadata.output_names != ("logits",):
                raise ValueError("semantic bundles require a logits output")
        elif metadata.task == "depth":
            if metadata.num_classes != 0 or metadata.class_names:
                raise ValueError("depth bundles must not declare classes")
            if metadata.score_encoding != "metric_depth":
                raise ValueError("depth bundles require metric_depth encoding")
            if metadata.output_names != ("depth",):
                raise ValueError("depth bundles require a depth output")
        elif metadata.task == "pose":
            if metadata.score_encoding != "heatmap_sigmoid":
                raise ValueError("pose bundles require sigmoid heatmap scores")
            if metadata.output_names != ("heatmaps",):
                raise ValueError("pose bundles require a heatmaps output")
            if metadata.task_options.get("num_keypoints") != metadata.num_classes:
                raise ValueError("pose num_keypoints must match num_classes")
        return metadata

    @classmethod
    def load(cls, bundle: Union[str, Path]) -> "ModelMetadata":
        return cls.from_dict(json.loads((Path(bundle) / "trhash.json").read_text()))

    def save(self, bundle: Union[str, Path]) -> Path:
        path = Path(bundle) / "trhash.json"
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


def metadata_from_checkpoint(backend, model_file: str = "model.onnx") -> ModelMetadata:
    task = str(getattr(backend, "task", getattr(backend.model, "vision_task", "detection")))
    config = getattr(backend.model, "config", None) or getattr(
        backend.model, "detector_config", None
    )
    if task == "classification":
        return ModelMetadata(
            format_version=4,
            task=task,
            model_file=model_file,
            image_size=config.image_size,
            num_classes=len(backend.names),
            class_names=tuple(backend.names),
            grid_sizes=(),
            reg_max=0,
            box_encoding="none",
            score_encoding="softmax",
            recommended_confidence=0.0,
            output_names=("logits",),
            resize_mode="stretch",
            task_options={},
        )
    if task == "semantic_segmentation":
        return ModelMetadata(
            format_version=4,
            task=task,
            model_file=model_file,
            image_size=config.image_size,
            num_classes=int(backend.model.num_classes),
            class_names=tuple(backend.names),
            grid_sizes=(),
            reg_max=0,
            box_encoding="none",
            score_encoding="per_pixel_softmax",
            recommended_confidence=0.0,
            output_names=("logits",),
            resize_mode="stretch",
            task_options={},
        )
    if task == "depth":
        return ModelMetadata(
            format_version=4,
            task=task,
            model_file=model_file,
            image_size=config.image_size,
            num_classes=0,
            class_names=(),
            grid_sizes=(),
            reg_max=0,
            box_encoding="none",
            score_encoding="metric_depth",
            recommended_confidence=0.0,
            output_names=("depth",),
            resize_mode="stretch",
            task_options={"max_depth": backend.model.max_depth},
        )
    if task == "pose":
        num_keypoints = int(backend.model.num_keypoints)
        return ModelMetadata(
            format_version=4,
            task=task,
            model_file=model_file,
            image_size=config.image_size,
            num_classes=num_keypoints,
            class_names=tuple(backend.names),
            grid_sizes=(),
            reg_max=0,
            box_encoding="none",
            score_encoding="heatmap_sigmoid",
            recommended_confidence=0.25,
            output_names=("heatmaps",),
            resize_mode="stretch",
            task_options={"num_keypoints": num_keypoints},
        )
    if task == "instance_segmentation":
        return ModelMetadata(
            format_version=4,
            task=task,
            model_file=model_file,
            image_size=config.image_size,
            num_classes=config.num_classes,
            class_names=tuple(backend.names),
            grid_sizes=tuple(config.grid_sizes),
            reg_max=config.reg_max,
            box_encoding="stride_ltrb_dfl",
            score_encoding="quality_class_sigmoid",
            recommended_confidence=float(
                backend.validation.get("best_confidence", 0.25)
            ),
            output_names=("predictions", "mask_coefficients", "prototypes"),
            resize_mode="letterbox",
            task_options={
                "num_prototypes": int(backend.model.num_prototypes),
                "mask_threshold": 0.5,
            },
        )
    if task != "detection":
        raise NotImplementedError(f"portable export is not implemented for task={task}")
    return ModelMetadata(
        format_version=4,
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
        output_names=("predictions",),
        task_options={},
    )
