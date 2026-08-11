from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("complexity")

from complexity.generative.detection import TRHashDetectorConfig  # noqa: E402
from complexity.generative.vision_tasks import (  # noqa: E402
    create_vision_model,
    save_vision_task_checkpoint,
)
from trhash import (  # noqa: E402
    ClassificationResult,
    DepthResult,
    SemanticSegmentationResult,
    Vision,
)


def _config() -> TRHashDetectorConfig:
    return TRHashDetectorConfig(
        image_size=32,
        patch_size=8,
        vision_hidden_size=32,
        vision_layers=1,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=16,
        vision_precision="fp32",
        num_classes=3,
    )


def test_framework_classification_checkpoint_to_portable_bundle(tmp_path: Path):
    config = _config()
    checkpoint = save_vision_task_checkpoint(
        create_vision_model("classification", config, num_classes=3),
        tmp_path / "checkpoint",
        task="classification",
        class_names=("cat", "dog", "bird"),
    )

    local = Vision(checkpoint, runtime="torch", device="cpu")
    local_result = local.predict(Image.new("RGB", (40, 20), "white"))
    bundle = local.export(format="torchscript", output=tmp_path / "bundle")
    portable_result = Vision(bundle, device="cpu").predict(
        Image.new("RGB", (40, 20), "white")
    )

    assert isinstance(local_result, ClassificationResult)
    assert isinstance(portable_result, ClassificationResult)
    assert portable_result.top1 == local_result.top1


def test_framework_semantic_checkpoint_to_portable_bundle(tmp_path: Path):
    checkpoint = save_vision_task_checkpoint(
        create_vision_model("semantic_segmentation", _config(), num_classes=3),
        tmp_path / "semantic-checkpoint",
        task="semantic_segmentation",
        class_names=("road", "person", "sky"),
    )

    local = Vision(checkpoint, runtime="torch", device="cpu")
    local_result = local.predict(Image.new("RGB", (40, 20), "white"))
    bundle = local.export(
        format="torchscript",
        output=tmp_path / "semantic-bundle",
    )
    portable_result = Vision(bundle, device="cpu").predict(
        Image.new("RGB", (40, 20), "white")
    )

    assert isinstance(local_result, SemanticSegmentationResult)
    assert isinstance(portable_result, SemanticSegmentationResult)
    assert portable_result.mask.size == (40, 20)
    assert list(portable_result.mask.get_flattened_data()) == list(
        local_result.mask.get_flattened_data()
    )


def test_framework_depth_checkpoint_to_portable_bundle(tmp_path: Path):
    checkpoint = save_vision_task_checkpoint(
        create_vision_model("depth", _config(), max_depth=80.0),
        tmp_path / "depth-checkpoint",
        task="depth",
    )

    local = Vision(checkpoint, runtime="torch", device="cpu")
    local_result = local.predict(Image.new("RGB", (40, 20), "white"))
    bundle = local.export(format="torchscript", output=tmp_path / "depth-bundle")
    portable_result = Vision(bundle, device="cpu").predict(
        Image.new("RGB", (40, 20), "white")
    )

    assert isinstance(local_result, DepthResult)
    assert isinstance(portable_result, DepthResult)
    assert portable_result.depth.size == (40, 20)
    assert list(portable_result.depth.get_flattened_data()) == pytest.approx(
        list(local_result.depth.get_flattened_data()),
        abs=1e-5,
    )
