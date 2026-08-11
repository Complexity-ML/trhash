"""Public dispatch for portable model exports."""

from __future__ import annotations

from .exporting import export_onnx, export_torchscript



def export_model(backend, *, format: str = "onnx", **options):
    exporters = {
        "onnx": export_onnx,
        "torchscript": export_torchscript,
    }
    try:
        exporter = exporters[format.casefold()]
    except KeyError as error:
        raise ValueError("format must be onnx or torchscript") from error
    if format.casefold() != "onnx":
        options.pop("opset", None)
    return exporter(backend, **options)


__all__ = ["export_model", "export_onnx", "export_torchscript"]
