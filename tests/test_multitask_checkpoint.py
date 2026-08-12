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
    InstanceSegmentationResult,
    OBBResult,
    PoseResult,
    SemanticSegmentationResult,
    Vision,
)
from trhash.metadata import CURRENT_FORMAT_VERSION, ModelMetadata  # noqa: E402


def _config() -> TRHashDetectorConfig:
    return TRHashDetectorConfig(
        image_size=32,
        patch_size=8,
        vision_hidden_size=32,
        vision_layers=3,
        vision_stage_depths=(1, 1, 1),
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=16,
        vision_precision="fp32",
        num_classes=3,
    )


def _assert_v5_bundle(bundle: Path) -> None:
    assert CURRENT_FORMAT_VERSION == 5
    assert ModelMetadata.load(bundle).format_version == CURRENT_FORMAT_VERSION


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
    _assert_v5_bundle(bundle)
    portable_result = Vision(bundle, device="cpu").predict(Image.new("RGB", (40, 20), "white"))

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
    _assert_v5_bundle(bundle)
    portable_result = Vision(bundle, device="cpu").predict(Image.new("RGB", (40, 20), "white"))

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
    _assert_v5_bundle(bundle)
    portable_result = Vision(bundle, device="cpu").predict(Image.new("RGB", (40, 20), "white"))

    assert isinstance(local_result, DepthResult)
    assert isinstance(portable_result, DepthResult)
    assert portable_result.depth.size == (40, 20)
    assert list(portable_result.depth.get_flattened_data()) == pytest.approx(
        list(local_result.depth.get_flattened_data()),
        abs=1e-5,
    )


def test_framework_pose_checkpoint_to_portable_bundle(tmp_path: Path):
    checkpoint = save_vision_task_checkpoint(
        create_vision_model("pose", _config(), num_keypoints=3),
        tmp_path / "pose-checkpoint",
        task="pose",
        class_names=("nose", "left_eye", "right_eye"),
    )

    local = Vision(checkpoint, runtime="torch", device="cpu")
    local_result = local.predict(Image.new("RGB", (40, 20), "white"))
    bundle = local.export(format="torchscript", output=tmp_path / "pose-bundle")
    _assert_v5_bundle(bundle)
    portable_result = Vision(bundle, device="cpu").predict(Image.new("RGB", (40, 20), "white"))

    assert isinstance(local_result, PoseResult)
    assert isinstance(portable_result, PoseResult)
    assert portable_result.names == ("nose", "left_eye", "right_eye")
    assert portable_result.keypoints == pytest.approx(local_result.keypoints, abs=1e-5)


def test_framework_instance_checkpoint_to_portable_bundle(tmp_path: Path):
    checkpoint = save_vision_task_checkpoint(
        create_vision_model("instance_segmentation", _config(), num_prototypes=4),
        tmp_path / "instance-checkpoint",
        task="instance_segmentation",
        class_names=("cat", "dog", "bird"),
    )

    local = Vision(checkpoint, runtime="torch", device="cpu")
    local_result = local.predict(
        Image.new("RGB", (40, 20), "white"),
        confidence=0.0,
    )
    bundle = local.export(format="torchscript", output=tmp_path / "instance-bundle")
    _assert_v5_bundle(bundle)
    portable_result = Vision(bundle, device="cpu").predict(
        Image.new("RGB", (40, 20), "white"),
        confidence=0.0,
    )

    assert isinstance(local_result, InstanceSegmentationResult)
    assert isinstance(portable_result, InstanceSegmentationResult)
    assert portable_result.labels == local_result.labels
    assert portable_result.boxes == pytest.approx(local_result.boxes, abs=1e-5)
    assert len(portable_result.masks) == len(local_result.masks)
    for actual, expected in zip(portable_result.masks, local_result.masks):
        assert actual.tobytes() == expected.tobytes()


def test_framework_obb_checkpoint_to_portable_bundle(tmp_path: Path):
    checkpoint = save_vision_task_checkpoint(
        create_vision_model("obb", _config()),
        tmp_path / "obb-checkpoint",
        task="obb",
        class_names=("car", "plane", "ship"),
    )

    local = Vision(checkpoint, runtime="torch", device="cpu")
    local_result = local.predict(
        Image.new("RGB", (40, 20), "white"),
        confidence=0.0,
    )
    bundle = local.export(format="torchscript", output=tmp_path / "obb-bundle")
    _assert_v5_bundle(bundle)
    portable_result = Vision(bundle, device="cpu").predict(
        Image.new("RGB", (40, 20), "white"),
        confidence=0.0,
    )

    assert isinstance(local_result, OBBResult)
    assert isinstance(portable_result, OBBResult)
    assert portable_result.labels == local_result.labels
    assert portable_result.boxes == pytest.approx(local_result.boxes, abs=1e-5)
