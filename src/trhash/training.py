"""Local fine-tuning orchestration, isolated from the public model facade."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

from .data import load_dataset, observed_class_count
from .training_options import architecture_arguments


class FineTuner:
    def __init__(self, backend) -> None:
        self.backend = backend

    def run(
        self,
        *,
        data: Union[str, Path],
        output: Union[str, Path] = "runs/train",
        epochs: int = 20,
        batch: int = 16,
        devices: int = 1,
        eval_batch: int = 0,
        eval_every: int = 5,
        eval_max_detections: int = 100,
        workers: int = 0,
        device: Optional[str] = None,
        lr: float = 1e-2,
        expert_lr_multiplier: float = 1.5,
        augmentation: str = "strong",
        seed: int = 42,
        resume: bool = False,
        extra_args: Sequence[str] = (),
    ) -> Path:
        if augmentation not in {"light", "strong"}:
            raise ValueError("augmentation must be light or strong")
        if eval_batch < 0:
            raise ValueError("eval_batch cannot be negative")
        if eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if eval_max_detections <= 0:
            raise ValueError("eval_max_detections must be positive")
        if devices <= 0:
            raise ValueError("devices must be positive")
        dataset = load_dataset(data)
        observed_classes = observed_class_count(dataset.train_labels)
        if observed_classes != len(dataset.names):
            raise ValueError(
                "dataset names and observed training class IDs disagree: "
                f"declared={len(dataset.names)}, observed={observed_classes}"
            )

        output_path = Path(output).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        class_names_path = output_path / "class_names.json"
        class_names_path.write_text(json.dumps(list(dataset.names), indent=2) + "\n")

        initialization = []
        if resume:
            training_state = self.backend.checkpoint / "training_state.pt"
            if not training_state.is_file():
                raise ValueError(
                    f"{self.backend.checkpoint} is weights-only and cannot be resumed exactly"
                )
            if tuple(self.backend.names) != tuple(dataset.names):
                raise ValueError("resume requires dataset class names in the original order")
            initialization.extend(("--resume", str(self.backend.checkpoint)))
        else:
            source_classes = {
                name.casefold(): index for index, name in enumerate(self.backend.names)
            }
            class_mapping = {
                target: source_classes[name.casefold()]
                for target, name in enumerate(dataset.names)
                if name.casefold() in source_classes
            }
            class_map_path = output_path / "class_map.json"
            class_map_path.write_text(json.dumps(class_mapping, indent=2) + "\n")
            initialization.extend(
                (
                    "--detector-checkpoint",
                    str(self.backend.checkpoint),
                    "--class-map",
                    str(class_map_path),
                )
            )

        selected_device = device or str(self.backend.device)
        if devices > 1 and not selected_device.startswith("cuda"):
            raise ValueError("multi-device training requires CUDA")
        launcher = (
            [
                sys.executable,
                "-u",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node",
                str(devices),
                "--module",
                "complexity.generative.detection.training",
            ]
            if devices > 1
            else [
                sys.executable,
                "-u",
                "-m",
                "complexity.generative.detection.training",
            ]
        )
        command = [
            *launcher,
            *initialization,
            "--yolo-images",
            str(dataset.train_images),
            "--yolo-labels",
            str(dataset.train_labels),
            "--output",
            str(output_path),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch),
            "--eval-batch-size",
            str(eval_batch),
            "--eval-every",
            str(eval_every),
            "--eval-max-detections",
            str(eval_max_detections),
            "--workers",
            str(workers),
            "--lr",
            str(lr),
            "--expert-lr-multiplier",
            str(expert_lr_multiplier),
            "--augmentation",
            augmentation,
            "--seed",
            str(seed),
            "--device",
            selected_device,
            *architecture_arguments(self.backend.model.config),
        ]
        if dataset.validation_images is not None and dataset.validation_labels is not None:
            command.extend(
                (
                    "--validation-yolo-images",
                    str(dataset.validation_images),
                    "--validation-yolo-labels",
                    str(dataset.validation_labels),
                )
            )
        command.extend(str(value) for value in extra_args)
        subprocess.run(command, check=True)

        candidates = [output_path / "best", *sorted(output_path.glob("step_*"), reverse=True)]
        checkpoint = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if checkpoint is None:
            raise RuntimeError(f"training produced no checkpoint in {output_path}")
        (checkpoint / "class_names.json").write_text(
            json.dumps(list(dataset.names), indent=2) + "\n"
        )
        return checkpoint
