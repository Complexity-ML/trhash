from pathlib import Path
from types import SimpleNamespace

import pytest

from trhash.training import FineTuner


def _dataset(root: Path) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "train" / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    config = root / "dataset.yaml"
    config.write_text("path: .\ntrain: images/train\nnames:\n  0: object\n")
    return config


def test_augmentation_is_explicit_finetuning_policy(monkeypatch, tmp_path: Path):
    backend = SimpleNamespace(
        names=("object",),
        checkpoint=tmp_path / "source",
        device="cpu",
        model=SimpleNamespace(
            config=SimpleNamespace(
                image_size=32,
                patch_size=8,
                vision_hidden_size=32,
                vision_layers=1,
                vision_heads=4,
                vision_num_experts=2,
                vision_top_k=1,
                vision_expert_width=16,
                assignment_top_k=5,
                reg_max=4,
                head_hidden_size=0,
                dfl_loss_weight=0.5,
                quality_focal_beta=2.0,
                box_loss_weight=5.0,
                quality_loss_weight=1.0,
                box_l1_weight=0.25,
                box_iou_weight=1.0,
                multi_scale=True,
                p2_head=True,
                dynamic_assignment=True,
                stal_enabled=True,
                progressive_loss_enabled=True,
            )
        ),
    )
    command = []

    def run(values, *, check):
        assert check
        command.extend(values)
        output = Path(values[values.index("--output") + 1])
        (output / "best").mkdir(parents=True)

    monkeypatch.setattr("trhash.training.subprocess.run", run)

    FineTuner(backend).run(
        data=_dataset(tmp_path / "data"),
        output=tmp_path / "output",
        augmentation="light",
    )

    assert command[command.index("--augmentation") + 1] == "light"


def test_unknown_augmentation_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="augmentation must be light or strong"):
        FineTuner(SimpleNamespace()).run(
            data=tmp_path / "unused.yaml",
            augmentation="mosaic-only",
        )


def test_resume_uses_exact_checkpoint_without_transfer_mapping(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    (checkpoint / "training_state.pt").touch()
    backend = SimpleNamespace(
        names=("object",),
        checkpoint=checkpoint,
        device="cpu",
        model=SimpleNamespace(
            config=SimpleNamespace(
                image_size=32,
                patch_size=8,
                vision_hidden_size=32,
                vision_layers=1,
                vision_heads=4,
                vision_num_experts=2,
                vision_top_k=1,
                vision_expert_width=16,
                assignment_top_k=5,
                reg_max=4,
                head_hidden_size=0,
                dfl_loss_weight=0.5,
                quality_focal_beta=2.0,
                box_loss_weight=5.0,
                quality_loss_weight=1.0,
                box_l1_weight=0.25,
                box_iou_weight=1.0,
                multi_scale=True,
                p2_head=True,
                dynamic_assignment=True,
                stal_enabled=True,
                progressive_loss_enabled=True,
            )
        ),
    )
    command = []

    def run(values, *, check):
        assert check
        command.extend(values)
        output = Path(values[values.index("--output") + 1])
        (output / "step_000010").mkdir(parents=True)

    monkeypatch.setattr("trhash.training.subprocess.run", run)
    FineTuner(backend).run(
        data=_dataset(tmp_path / "data"),
        output=tmp_path / "output",
        resume=True,
    )

    assert command[command.index("--resume") + 1] == str(checkpoint)
    assert "--detector-checkpoint" not in command
    assert "--class-map" not in command


def test_resume_rejects_weights_only_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    backend = SimpleNamespace(
        names=("object",),
        checkpoint=checkpoint,
        device="cpu",
    )
    with pytest.raises(ValueError, match="weights-only"):
        FineTuner(backend).run(
            data=_dataset(tmp_path / "data"),
            output=tmp_path / "output",
            resume=True,
        )
