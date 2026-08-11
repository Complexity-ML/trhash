"""Export a training checkpoint into a framework-independent ONNX bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .metadata import metadata_from_checkpoint


def export_onnx(
    backend,
    *,
    output: Union[str, Path] = "runs/export",
    opset: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("ONNX export requires PyTorch") from error

    class ExportDetector(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, pixel_values):
            one_to_many, one_to_one = self.detector.forward_predictions(pixel_values)
            return one_to_one if one_to_one is not None else one_to_many

    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "model.onnx"
    detector = ExportDetector(backend.model.to("cpu").eval())
    size = backend.model.config.image_size
    example = torch.zeros(1, 3, size, size, dtype=torch.float32)
    dynamic_axes = (
        {"pixel_values": {0: "batch"}, "predictions": {0: "batch"}}
        if dynamic_batch
        else None
    )
    with torch.inference_mode():
        torch.onnx.export(
            detector,
            example,
            str(model_path),
            input_names=("pixel_values",),
            output_names=("predictions",),
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    metadata_from_checkpoint(backend).save(output_path)
    return output_path
