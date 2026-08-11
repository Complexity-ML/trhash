"""TorchScript export with raw-output parity validation."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from ..metadata import metadata_from_checkpoint
from .common import assert_parity, parity_inputs, prepare_export


def export_torchscript(
    backend,
    *,
    output: Union[str, Path] = "runs/export",
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    output_path, detector, example = prepare_export(backend, output)
    model_path = output_path / "model.torchscript"
    with torch.inference_mode():
        # Keep parameters/buffers movable; freezing turns them into CPU constants
        # that cannot subsequently be transferred to MPS.
        exported = torch.jit.trace(detector, example, strict=True).eval()
        torch.jit.save(exported, str(model_path))
    if verify:
        loaded = torch.jit.load(str(model_path), map_location="cpu").eval()
        assert_parity(
            detector,
            loaded,
            parity_inputs(example, dynamic_batch=dynamic_batch),
            atol=1e-5,
            rtol=1e-5,
        )
    metadata_from_checkpoint(backend, model_file=model_path.name).save(output_path)
    return output_path
