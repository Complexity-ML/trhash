from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from trhash import (  # noqa: E402
    ClassificationResult,
    DepthResult,
    InstanceSegmentationResult,
    OBBResult,
    PoseResult,
    SemanticSegmentationResult,
    Vision,
)
from trhash.exporter import export_model  # noqa: E402
from trhash.exporting.common import ExportVisionModel  # noqa: E402
from trhash.metadata import ModelMetadata, metadata_from_checkpoint  # noqa: E402


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
            end_to_end=False,
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


class TinyNMSFreeDetector(TinyDetector):
    def __init__(self):
        super().__init__()
        self.config.end_to_end = True

    def forward_branches(self, pixel_values):
        one_to_many = self.forward_predictions(pixel_values)
        one_to_one = one_to_many.clone()
        one_to_one[..., -2:] = torch.tensor([[-4.0, 4.0]])
        return one_to_many, one_to_one


def _nms_free_backend():
    return SimpleNamespace(
        model=TinyNMSFreeDetector(),
        names=("cat", "dog"),
        validation={"best_confidence": 0.2},
    )


def test_detection_export_defaults_to_one_to_one_nms_free_output():
    backend = _nms_free_backend()
    wrapper = ExportVisionModel(backend.model, "detection")
    pixels = torch.zeros(1, 3, 16, 16)

    output = wrapper(pixels)
    _, expected = backend.model.forward_branches(pixels)
    metadata = metadata_from_checkpoint(backend)

    torch.testing.assert_close(output, expected)
    assert metadata.task_options == {"postprocess": "nms_free"}


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("base", torch.tensor([[1.0, 3.0, -1.0]]))
        self.detector_config = SimpleNamespace(image_size=16)

    def forward(self, pixel_values):
        adjustment = pixel_values.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        return {"logits": self.base.expand(pixel_values.shape[0], -1) + adjustment * 0.01}


def _classification_backend():
    return SimpleNamespace(
        model=TinyClassifier(),
        task="classification",
        names=("cat", "dog", "bird"),
        validation={},
    )


class TinySemanticSegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.detector_config = SimpleNamespace(image_size=16)
        self.num_classes = 2

    def forward(self, pixel_values):
        first = pixel_values[:, :1]
        return {"logits": torch.cat((first, -first), dim=1)}


def _semantic_backend():
    return SimpleNamespace(
        model=TinySemanticSegmenter(),
        task="semantic_segmentation",
        names=("light", "dark"),
        validation={},
    )


class TinyDepthEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.detector_config = SimpleNamespace(image_size=16)
        self.max_depth = 80.0

    def forward(self, pixel_values):
        return {"depth": pixel_values[:, :1].abs() + 1.0}


def _depth_backend():
    return SimpleNamespace(
        model=TinyDepthEstimator(),
        task="depth",
        names=(),
        validation={},
    )


class TinyPoseEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.detector_config = SimpleNamespace(image_size=16)
        self.num_keypoints = 2

    def forward(self, pixel_values):
        first = pixel_values[:, :1]
        second = torch.flip(first, dims=(-1,))
        return {"heatmaps": torch.sigmoid(torch.cat((first, second), dim=1))}


def _pose_backend():
    return SimpleNamespace(
        model=TinyPoseEstimator(),
        task="pose",
        names=("nose", "tail"),
        validation={},
    )


class TinyInstanceSegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            image_size=16,
            num_classes=2,
            grid_sizes=(1,),
            reg_max=0,
        )
        self.num_prototypes = 2
        self.register_buffer(
            "raw",
            torch.tensor([[[0.0, 0.0, 0.0, 0.0, 4.0, -4.0]]]),
        )
        self.register_buffer("coefficients", torch.tensor([[[5.0, -5.0]]]))
        self.register_buffer(
            "prototypes",
            torch.stack((torch.ones(4, 4), -torch.ones(4, 4)))[None],
        )

    def forward_instance(self, pixel_values):
        batch = pixel_values.shape[0]
        adjustment = pixel_values.mean(dim=(1, 2, 3), keepdim=True)
        return {
            "raw": self.raw.expand(batch, -1, -1) + adjustment.reshape(batch, 1, 1) * 0.01,
            "mask_coefficients": self.coefficients.expand(batch, -1, -1),
            "prototypes": self.prototypes.expand(batch, -1, -1, -1),
        }


def _instance_backend():
    return SimpleNamespace(
        model=TinyInstanceSegmenter(),
        task="instance_segmentation",
        names=("cat", "dog"),
        validation={"best_confidence": 0.25},
    )


class TinyOBBDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            image_size=16,
            num_classes=2,
            grid_sizes=(1,),
            reg_max=0,
        )
        self.register_buffer(
            "raw",
            torch.tensor([[[0.0, 0.0, 0.0, 0.0, 4.0, -4.0]]]),
        )
        self.register_buffer("angles", torch.tensor([[0.4]]))

    def forward_obb(self, pixel_values):
        batch = pixel_values.shape[0]
        adjustment = pixel_values.mean(dim=(1, 2, 3), keepdim=True)
        return {
            "raw": self.raw.expand(batch, -1, -1) + adjustment.reshape(batch, 1, 1) * 0.01,
            "angles": self.angles.expand(batch, -1) + adjustment.reshape(batch, 1) * 0.001,
        }


def _obb_backend():
    return SimpleNamespace(
        model=TinyOBBDetector(),
        task="obb",
        names=("car", "plane"),
        validation={"best_confidence": 0.25},
    )


class ExportableBackend:
    def __init__(self):
        values = _backend()
        self.model = values.model
        self.names = values.names
        self.validation = values.validation

    def export(self, **options):
        return export_model(self, **options)


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
    assert set(results[0].speed) == {"preprocess", "inference", "postprocess"}


def test_onnx_export_parity_and_auto_runtime(tmp_path: Path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    bundle = export_model(_backend(), format="onnx", output=tmp_path / "onnx")

    model = Vision(bundle)
    result = model.predict(Image.new("RGB", (16, 16), "white"))

    assert type(model.backend).__name__ == "OnnxBackend"
    assert result.labels == [0]


@pytest.mark.parametrize("format", ("onnx", "torchscript", "coreml"))
def test_classification_export_and_runtime(tmp_path: Path, format: str):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "coreml":
        pytest.importorskip("coremltools")
    bundle = export_model(
        _classification_backend(),
        format=format,
        output=tmp_path / format,
        **({"precision": "fp32"} if format == "coreml" else {}),
    )

    metadata = ModelMetadata.load(bundle)
    result = Vision(bundle).predict(Image.new("RGB", (24, 12), "white"))

    assert metadata.task == "classification"
    assert metadata.output_names == ("logits",)
    assert isinstance(result, ClassificationResult)
    assert result.top1 == 1
    assert result.names[result.top1] == "dog"
    assert sum(result.scores) == pytest.approx(1.0)
    assert result.to_dict()["task"] == "classification"


@pytest.mark.parametrize("format", ("onnx", "torchscript", "coreml"))
def test_semantic_export_and_runtime(tmp_path: Path, format: str):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "coreml":
        pytest.importorskip("coremltools")
    bundle = export_model(
        _semantic_backend(),
        format=format,
        output=tmp_path / format,
        **({"precision": "fp32"} if format == "coreml" else {}),
    )

    metadata = ModelMetadata.load(bundle)
    result = Vision(bundle).predict(Image.new("RGB", (24, 12), "white"))

    assert metadata.task == "semantic_segmentation"
    assert metadata.output_names == ("logits",)
    assert isinstance(result, SemanticSegmentationResult)
    assert result.mask.size == (24, 12)
    assert result.labels == (0,)
    assert result.to_dict()["task"] == "semantic_segmentation"


@pytest.mark.parametrize("format", ("onnx", "torchscript", "coreml"))
def test_depth_export_and_runtime(tmp_path: Path, format: str):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "coreml":
        pytest.importorskip("coremltools")
    bundle = export_model(
        _depth_backend(),
        format=format,
        output=tmp_path / format,
        **({"precision": "fp32"} if format == "coreml" else {}),
    )

    metadata = ModelMetadata.load(bundle)
    result = Vision(bundle).predict(Image.new("RGB", (24, 12), "white"))

    assert metadata.task == "depth"
    assert metadata.output_names == ("depth",)
    assert isinstance(result, DepthResult)
    assert result.depth.size == (24, 12)
    assert result.min_depth == pytest.approx(2.0)
    assert result.max_depth == pytest.approx(2.0)
    assert result.to_dict()["task"] == "depth"


@pytest.mark.parametrize("format", ("onnx", "torchscript", "coreml"))
def test_pose_export_and_runtime(tmp_path: Path, format: str):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "coreml":
        pytest.importorskip("coremltools")
    bundle = export_model(
        _pose_backend(),
        format=format,
        output=tmp_path / format,
        **({"precision": "fp32"} if format == "coreml" else {}),
    )

    metadata = ModelMetadata.load(bundle)
    result = Vision(bundle).predict(Image.new("RGB", (24, 12), "white"))

    assert metadata.task == "pose"
    assert metadata.output_names == ("heatmaps",)
    assert metadata.task_options == {"num_keypoints": 2}
    assert isinstance(result, PoseResult)
    assert result.names == ("nose", "tail")
    assert len(result.keypoints) == 2
    assert all(0.0 <= x <= 24.0 and 0.0 <= y <= 12.0 for x, y, _ in result.keypoints)
    assert result.to_dict()["task"] == "pose"


@pytest.mark.parametrize("format", ("onnx", "torchscript", "coreml"))
def test_instance_export_and_runtime(tmp_path: Path, format: str):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "coreml":
        pytest.importorskip("coremltools")
    bundle = export_model(
        _instance_backend(),
        format=format,
        output=tmp_path / format,
        **({"precision": "fp32"} if format == "coreml" else {}),
    )

    metadata = ModelMetadata.load(bundle)
    result = Vision(bundle).predict(Image.new("RGB", (24, 12), "white"))

    assert metadata.task == "instance_segmentation"
    assert metadata.output_names == (
        "predictions",
        "mask_coefficients",
        "prototypes",
    )
    assert metadata.task_options["num_prototypes"] == 2
    assert isinstance(result, InstanceSegmentationResult)
    assert result.labels == [0]
    assert result.masks[0].size == (24, 12)
    assert result.masks[0].getextrema() == (255, 255)
    assert result.to_dict()["task"] == "instance_segmentation"


@pytest.mark.parametrize("format", ("onnx", "torchscript", "coreml"))
def test_obb_export_and_runtime(tmp_path: Path, format: str):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "coreml":
        pytest.importorskip("coremltools")
    bundle = export_model(
        _obb_backend(),
        format=format,
        output=tmp_path / format,
        **({"precision": "fp32"} if format == "coreml" else {}),
    )

    metadata = ModelMetadata.load(bundle)
    result = Vision(bundle).predict(Image.new("RGB", (24, 12), "white"))

    assert metadata.task == "obb"
    assert metadata.output_names == ("predictions", "angles")
    assert metadata.task_options == {"angle_unit": "radians"}
    assert isinstance(result, OBBResult)
    assert result.labels == [0]
    assert result.boxes[0][4] == pytest.approx(0.400447, abs=1e-4)
    assert result.to_dict()["task"] == "obb"


def test_coreml_export_parity_and_auto_runtime(tmp_path: Path):
    pytest.importorskip("coremltools")
    bundle = export_model(
        _backend(),
        format="coreml",
        output=tmp_path / "coreml",
        precision="fp32",
    )

    model = Vision(bundle, device="cpu")
    results = model.predict(
        [Image.new("RGB", (16, 16), "white"), Image.new("RGB", (16, 16), "black")],
        batch=2,
    )

    assert type(model.backend).__name__ == "CoreMLBackend"
    assert len(results) == 2
    assert all(result.labels == [0] for result in results)


def test_tensorrt_export_requires_nvidia_runtime(tmp_path: Path):
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match=r"trhash\[tensorrt\]"):
            export_model(_backend(), format="tensorrt", output=tmp_path / "tensorrt")
    else:
        pytest.skip("TensorRT runtime is installed; CUDA integration test owns this path")


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


def test_benchmark_compares_both_exported_formats(tmp_path: Path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = Vision.__new__(Vision)
    model.backend = ExportableBackend()

    report = model.benchmark(
        Image.new("RGB", (16, 16), "white"),
        formats=("onnx", "torchscript"),
        output=tmp_path / "benchmark",
        warmup=1,
        runs=2,
        batch=2,
        device="cpu",
    )

    assert {entry.format for entry in report.entries} == {"onnx", "torchscript"}
    assert report.fastest in {"onnx", "torchscript"}
    assert all(entry.latency_ms > 0 for entry in report.entries)
    assert all(entry.bundle_size_mb > 0 for entry in report.entries)
