"""ONNX export with raw-output parity validation."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from ..metadata import metadata_from_checkpoint
from .common import parity_inputs, prepare_export, tensor_outputs


def _verify_onnx(model_path: Path, reference, inputs) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("ONNX parity validation requires onnx and onnxruntime") from error
    onnx.checker.check_model(onnx.load(str(model_path)))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        for values in inputs:
            expected = tensor_outputs(reference(values))
            actual = session.run(reference.output_names, {"pixel_values": values.numpy()})
            if len(actual) != len(expected):
                raise AssertionError("ONNX model returned the wrong number of outputs")
            for actual_array, expected_tensor in zip(actual, expected):
                np.testing.assert_allclose(
                    actual_array,
                    expected_tensor.detach().cpu().numpy(),
                    atol=1e-5,
                    rtol=1e-4,
                )


def export_onnx(
    backend,
    *,
    output: Union[str, Path] = "runs/export",
    opset: int = 18,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    output_path, detector, example = prepare_export(backend, output)
    model_path = output_path / "model.onnx"
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"pixel_values": {0: "batch"}}
        dynamic_axes.update({name: {0: "batch"} for name in detector.output_names})
    with torch.inference_mode():
        torch.onnx.export(
            detector,
            example,
            str(model_path),
            input_names=("pixel_values",),
            output_names=detector.output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    if verify:
        _verify_onnx(model_path, detector, parity_inputs(example, dynamic_batch=dynamic_batch))
    metadata_from_checkpoint(backend, model_file=model_path.name).save(output_path)
    return output_path
