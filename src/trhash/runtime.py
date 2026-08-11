"""Lazy loading and checkpoint resolution for the optional local runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def imports():
    try:
        import torch
        from complexity.generative.detection import (
            load_detector_checkpoint,
            preprocess_detector_image,
            restore_detector_boxes,
        )
        from complexity.generative.vision_tasks import load_vision_task_checkpoint
    except ImportError as error:
        raise RuntimeError(
            'local execution requires PyTorch and `pip install "trhash[local]"`'
        ) from error
    return (
        torch,
        load_detector_checkpoint,
        preprocess_detector_image,
        restore_detector_boxes,
        load_vision_task_checkpoint,
    )


def resolve_device(torch, requested: Optional[str]):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint(
    model: Union[str, Path],
    *,
    revision: Optional[str],
    token: Optional[str],
) -> Path:
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            'Hub downloads require `pip install "trhash[local]"`'
        ) from error
    return Path(
        snapshot_download(
            repo_id=str(model),
            revision=revision,
            token=token,
            allow_patterns=(
                "config.json",
                "model.safetensors",
                "class_names.json",
                "validation.json",
                "vision_task.json",
            ),
        )
    )
