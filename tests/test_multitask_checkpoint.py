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
from trhash import ClassificationResult, Vision  # noqa: E402


def test_framework_classification_checkpoint_to_portable_bundle(tmp_path: Path):
    config = TRHashDetectorConfig(
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
