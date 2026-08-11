from pathlib import Path
from types import SimpleNamespace

import pytest

from trhash.tuning import run_tuning


SPACE = {
    "lr": (0.005,),
    "expert_lr_multiplier": (1.5,),
    "augmentation": ("strong",),
}


class BaseModel:
    def __init__(self):
        self.backend = SimpleNamespace(task="detection", train=object())
        self.train_calls = []

    def train(self, **options):
        self.train_calls.append(options)
        checkpoint = Path(options["output"]) / "best"
        checkpoint.mkdir(parents=True, exist_ok=True)
        return checkpoint


def _candidate_type(resume_calls):
    class Candidate:
        def __init__(self, checkpoint, **_options):
            self.checkpoint = Path(checkpoint)

        def train(self, **options):
            resume_calls.append(options)
            checkpoint = Path(options["output"]) / "best"
            checkpoint.mkdir(parents=True, exist_ok=True)
            return checkpoint

        def val(self, **_options):
            trial = int(self.checkpoint.parent.name.rsplit("_", 1)[-1])
            metrics = {
                "map50": 0.5 + trial * 0.1,
                "precision": 0.6,
                "recall": 0.7,
                "f1": 0.64 + trial * 0.01,
                "best_confidence": 0.25,
                "images": 2,
                "targets": 2,
                "predictions": 2,
                "per_class_ap50": {"object": 0.5 + trial * 0.1},
            }
            return SimpleNamespace(to_dict=lambda: metrics)

        def close(self):
            pass

    return Candidate


def test_tune_persists_results_and_skips_completed_trials(monkeypatch, tmp_path: Path):
    model = BaseModel()
    resume_calls = []
    monkeypatch.setattr("trhash.model.Vision", _candidate_type(resume_calls))
    space = {
        "lr": (0.002, 0.01),
        "expert_lr_multiplier": (1.5,),
        "augmentation": ("strong",),
    }

    report = run_tuning(
        model,
        data=tmp_path / "dataset.yaml",
        output=tmp_path / "tune",
        iterations=2,
        epochs=1,
        space=space,
    )

    assert len(model.train_calls) == 2
    assert report.best.index == 1
    assert Path(report.best.checkpoint).name == "best"
    assert (tmp_path / "tune" / "tune_plan.json").is_file()
    assert (tmp_path / "tune" / "tune_report.json").is_file()

    model.train_calls.clear()
    resumed = run_tuning(
        model,
        data=tmp_path / "dataset.yaml",
        output=tmp_path / "tune",
        iterations=2,
        epochs=1,
        space=space,
        resume=True,
    )

    assert model.train_calls == []
    assert resumed.to_dict() == report.to_dict()


def test_tune_resumes_interrupted_training_state(monkeypatch, tmp_path: Path):
    class InterruptedModel(BaseModel):
        def train(self, **options):
            checkpoint = Path(options["output"]) / "step_000010"
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "training_state.pt").touch()
            raise KeyboardInterrupt

    model = InterruptedModel()
    output = tmp_path / "tune"
    with pytest.raises(KeyboardInterrupt):
        run_tuning(
            model,
            data=tmp_path / "dataset.yaml",
            output=output,
            iterations=1,
            epochs=2,
            space=SPACE,
        )

    resume_calls = []
    monkeypatch.setattr("trhash.model.Vision", _candidate_type(resume_calls))
    report = run_tuning(
        model,
        data=tmp_path / "dataset.yaml",
        output=output,
        iterations=1,
        epochs=2,
        space=SPACE,
        resume=True,
    )

    assert report.best.status == "completed"
    assert len(resume_calls) == 1
    assert resume_calls[0]["resume"] is True


def test_tune_rejects_unknown_or_oversized_search_spaces(tmp_path: Path):
    model = BaseModel()
    with pytest.raises(ValueError, match="unsupported tuning parameters"):
        run_tuning(
            model,
            data=tmp_path / "dataset.yaml",
            output=tmp_path / "unknown",
            iterations=1,
            space={"momentum": (0.9,)},
        )
    with pytest.raises(ValueError, match="iterations must be between"):
        run_tuning(
            model,
            data=tmp_path / "dataset.yaml",
            output=tmp_path / "too-many",
            iterations=2,
            space=SPACE,
        )
