"""Deterministic, resumable fine-tuning search for detection checkpoints."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


DEFAULT_SPACE = {
    "lr": (0.002, 0.005, 0.01),
    "expert_lr_multiplier": (1.0, 1.5, 2.0),
    "augmentation": ("light", "strong"),
}


@dataclass(frozen=True)
class TuneTrial:
    index: int
    parameters: Dict[str, Any]
    status: str
    checkpoint: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class TuneReport:
    output: str
    trials: tuple[TuneTrial, ...]

    @property
    def completed(self) -> tuple[TuneTrial, ...]:
        return tuple(trial for trial in self.trials if trial.status == "completed")

    @property
    def best(self) -> TuneTrial:
        if not self.completed:
            raise RuntimeError("tuning produced no completed trial")
        return max(
            self.completed,
            key=lambda trial: (
                float(trial.metrics["map50"]),
                float(trial.metrics["f1"]),
            ),
        )

    def to_dict(self) -> dict:
        payload = {
            "output": self.output,
            "trials": [asdict(trial) for trial in self.trials],
        }
        if self.completed:
            payload["best"] = asdict(self.best)
        return payload


def _normalized_space(space: Optional[Dict[str, Sequence[Any]]]) -> Dict[str, tuple[Any, ...]]:
    values = DEFAULT_SPACE if space is None else space
    unknown = sorted(set(values) - set(DEFAULT_SPACE))
    if unknown:
        raise ValueError(f"unsupported tuning parameters: {unknown}")
    normalized = {}
    for name in DEFAULT_SPACE:
        candidates = tuple(values.get(name, DEFAULT_SPACE[name]))
        if not candidates:
            raise ValueError(f"tuning space for {name} must not be empty")
        if name in {"lr", "expert_lr_multiplier"}:
            candidates = tuple(float(value) for value in candidates)
            if any(value <= 0 for value in candidates):
                raise ValueError(f"tuning values for {name} must be positive")
        else:
            candidates = tuple(str(value) for value in candidates)
            if any(value not in {"light", "strong"} for value in candidates):
                raise ValueError("augmentation candidates must be light or strong")
        normalized[name] = candidates
    return normalized


def _trial_plan(
    space: Dict[str, tuple[Any, ...]],
    iterations: int,
    seed: int,
) -> list[Dict[str, Any]]:
    names = tuple(space)
    combinations = [
        dict(zip(names, values))
        for values in itertools.product(*(space[name] for name in names))
    ]
    if iterations <= 0 or iterations > len(combinations):
        raise ValueError(
            f"iterations must be between 1 and {len(combinations)} for this space"
        )
    random.Random(seed).shuffle(combinations)
    return combinations[:iterations]


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _load_trial(path: Path, index: int, parameters: Dict[str, Any]) -> TuneTrial:
    if not path.is_file():
        return TuneTrial(index=index, parameters=parameters, status="pending")
    payload = json.loads(path.read_text())
    if payload.get("index") != index or payload.get("parameters") != parameters:
        raise ValueError(f"tuning trial metadata mismatch at {path}")
    return TuneTrial(**payload)


def _latest_resumable_checkpoint(directory: Path) -> Optional[Path]:
    candidates = sorted(directory.glob("step_*"), reverse=True)
    return next(
        (
            candidate
            for candidate in candidates
            if (candidate / "training_state.pt").is_file()
        ),
        None,
    )


def run_tuning(
    model,
    *,
    data: Union[str, Path],
    output: Union[str, Path] = "runs/tune",
    iterations: int = 10,
    epochs: int = 10,
    batch: int = 16,
    workers: int = 0,
    device: Optional[str] = None,
    seed: int = 42,
    space: Optional[Dict[str, Sequence[Any]]] = None,
    resume: bool = False,
    max_images: Optional[int] = None,
) -> TuneReport:
    """Train and validate a fixed search plan, resuming unfinished trials exactly."""

    backend = model.backend
    metadata = getattr(backend, "metadata", None)
    task = getattr(metadata, "task", getattr(backend, "task", "detection"))
    if task != "detection" or not hasattr(backend, "train"):
        raise ValueError("tune() requires a local PyTorch detection checkpoint")
    if epochs <= 0 or batch <= 0 or workers < 0:
        raise ValueError("epochs and batch must be positive; workers must be non-negative")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")

    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    normalized_space = _normalized_space(space)
    parameters = _trial_plan(normalized_space, iterations, seed)
    plan = {
        "format_version": 1,
        "model": str(
            getattr(backend, "checkpoint", getattr(backend, "model_id", "unknown"))
        ),
        "data": str(Path(data).expanduser().resolve()),
        "iterations": iterations,
        "epochs": epochs,
        "batch": batch,
        "workers": workers,
        "seed": seed,
        "max_images": max_images,
        "space": {name: list(values) for name, values in normalized_space.items()},
        "trials": parameters,
    }
    plan_path = output_path / "tune_plan.json"
    if plan_path.exists():
        if not resume:
            raise ValueError(f"{output_path} already contains a tuning plan; use resume=True")
        existing = json.loads(plan_path.read_text())
        if existing != plan:
            raise ValueError("resume options do not match the saved tuning plan")
    else:
        _write_json(plan_path, plan)

    from .model import Vision

    trials = []
    for index, trial_parameters in enumerate(parameters):
        trial_directory = output_path / f"trial_{index:03d}"
        trial_directory.mkdir(parents=True, exist_ok=True)
        state_path = trial_directory / "trial.json"
        previous = _load_trial(state_path, index, trial_parameters)
        if previous.status == "completed":
            trials.append(previous)
            continue
        running = TuneTrial(index=index, parameters=trial_parameters, status="running")
        _write_json(state_path, asdict(running))
        candidate_model = None
        try:
            resumable = _latest_resumable_checkpoint(trial_directory) if resume else None
            if resumable is not None:
                candidate_model = Vision(resumable, runtime="torch", device=device)
                checkpoint = candidate_model.train(
                    data=data,
                    output=trial_directory,
                    epochs=epochs,
                    batch=batch,
                    workers=workers,
                    device=device,
                    seed=seed,
                    resume=True,
                    **trial_parameters,
                )
            else:
                checkpoint = model.train(
                    data=data,
                    output=trial_directory,
                    epochs=epochs,
                    batch=batch,
                    workers=workers,
                    device=device,
                    seed=seed,
                    **trial_parameters,
                )
            if candidate_model is not None:
                candidate_model.close()
            candidate_model = Vision(checkpoint, runtime="torch", device=device)
            metrics = candidate_model.val(
                data=data,
                batch=batch,
                max_images=max_images,
            ).to_dict()
            completed = TuneTrial(
                index=index,
                parameters=trial_parameters,
                status="completed",
                checkpoint=str(checkpoint),
                metrics=metrics,
            )
            _write_json(state_path, asdict(completed))
            trials.append(completed)
        except Exception as error:
            failed = TuneTrial(
                index=index,
                parameters=trial_parameters,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
            _write_json(state_path, asdict(failed))
            trials.append(failed)
        finally:
            if candidate_model is not None:
                candidate_model.close()
        report = TuneReport(str(output_path), tuple(trials))
        _write_json(output_path / "tune_report.json", report.to_dict())

    report = TuneReport(str(output_path), tuple(trials))
    if not report.completed:
        raise RuntimeError(f"all tuning trials failed; inspect {output_path}")
    _write_json(output_path / "tune_report.json", report.to_dict())
    return report
