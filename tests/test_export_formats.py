from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from trhash import Vision  # noqa: E402
from trhash.exporter import export_model  # noqa: E402
from trhash.metadata import ModelMetadata  # noqa: E402


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "base",
            torch.tensor([[[0.0, 0.0, 0.0, 0.0, 4.0, -4.0]]]),
        )
        self.config = SimpleNamespace(
            image_size=16,
            num_classes=2,
            grid_sizes=(1,),
            reg_max=0,
        )

    def forward_predictions(self, pixel_values):
        adjustment = pixel_values.mean(dim=(1, 2, 3), keepdim=True).reshape(-1, 1, 1)
        return self.base.expand(pixel_values.shape[0], -1, -1) + adjustment * 0.01


def _backend():
    return SimpleNamespace(
        model=TinyDetector(),
        names=("cat", "dog"),
        validation={"best_confidence": 0.25},
    )


def test_torchscript_export_parity_and_auto_runtime(tmp_path: Path):
    bundle = export_model(_backend(), format="torchscript", output=tmp_path / "torchscript")

    metadata = ModelMetadata.load(bundle)
    assert metadata.model_file == "model.torchscript"
    model = Vision(bundle)
    results = model.predict(
        [Image.new("RGB", (16, 16), "white"), Image.new("RGB", (16, 16), "black")],
        batch=2,
    )

    assert type(model.backend).__name__ == "TorchScriptBackend"
    assert len(results) == 2
    assert all(result.labels == [0] for result in results)


def test_onnx_export_parity_and_auto_runtime(tmp_path: Path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    bundle = export_model(_backend(), format="onnx", output=tmp_path / "onnx")

    model = Vision(bundle)
    result = model.predict(Image.new("RGB", (16, 16), "white"))

    assert type(model.backend).__name__ == "OnnxBackend"
    assert result.labels == [0]


def test_runtime_rejects_format_mismatch(tmp_path: Path):
    bundle = export_model(_backend(), format="torchscript", output=tmp_path / "torchscript")

    with pytest.raises(ValueError, match="bundle contains torchscript"):
        Vision(bundle, runtime="onnx")


def test_torchscript_raw_output_matches_checkpoint(tmp_path: Path):
    backend = _backend()
    bundle = export_model(backend, format="torchscript", output=tmp_path / "torchscript")
    exported = torch.jit.load(str(bundle / "model.torchscript"))
    pixels = torch.from_numpy(np.random.default_rng(4).normal(size=(2, 3, 16, 16)).astype("float32"))

    torch.testing.assert_close(exported(pixels), backend.model.forward_predictions(pixels))
