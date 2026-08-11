"""Native Core ML Program export with prediction parity validation."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from ..metadata import metadata_from_checkpoint
from .common import parity_inputs, prepare_export


def export_coreml(
    backend,
    *,
    output: Union[str, Path] = "runs/export",
    dynamic_batch: bool = False,
    precision: str = "fp16",
    verify: bool = True,
) -> Path:
    try:
        import coremltools as ct
    except ImportError as error:
        raise RuntimeError('CoreML export requires `pip install "trhash[coreml]"`') from error
    if precision not in {"fp16", "fp32"}:
        raise ValueError("CoreML precision must be fp16 or fp32")
    if dynamic_batch:
        raise ValueError("CoreML export currently uses fixed batch 1; the SDK splits batches")

    output_path, detector, example = prepare_export(backend, output)
    model_path = output_path / "model.mlpackage"
    exported = torch.export.export(detector, (example,), strict=False).run_decompositions({})
    converted = ct.convert(
        exported,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS13,
    )
    converted.save(str(model_path))
    if verify:
        tolerance = 3e-2 if precision == "fp16" else 1e-4
        spec = converted.get_spec()
        input_name = spec.description.input[0].name
        output_name = spec.description.output[0].name
        with torch.inference_mode():
            for values in parity_inputs(example, dynamic_batch=False):
                expected = detector(values).cpu().numpy()
                actual = converted.predict({input_name: values.numpy()})[output_name]
                np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    metadata_from_checkpoint(backend, model_file=model_path.name).save(output_path)
    return output_path
