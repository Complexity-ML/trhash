"""Public dispatch for portable model exports."""

from __future__ import annotations

from .exporting import export_coreml, export_onnx, export_tensorrt, export_torchscript



def export_model(backend, *, format: str = "onnx", **options):
    exporters = {
        "onnx": export_onnx,
        "torchscript": export_torchscript,
        "coreml": export_coreml,
        "tensorrt": export_tensorrt,
    }
    try:
        exporter = exporters[format.casefold()]
    except KeyError as error:
        raise ValueError("format must be onnx, torchscript, coreml, or tensorrt") from error
    if format.casefold() != "onnx":
        options.pop("opset", None)
    return exporter(backend, **options)


__all__ = [
    "export_coreml",
    "export_model",
    "export_onnx",
    "export_tensorrt",
    "export_torchscript",
]
