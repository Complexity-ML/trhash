"""Shared preparation and parity checks for portable exports."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Union

import torch


class ExportDetector(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, pixel_values):
        return self.detector.forward_predictions(pixel_values)


def prepare_export(backend, output: Union[str, Path]):
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    detector = ExportDetector(copy.deepcopy(backend.model).to("cpu").eval())
    size = backend.model.config.image_size
    example = torch.zeros(1, 3, size, size, dtype=torch.float32)
    return output_path, detector, example


def parity_inputs(example: torch.Tensor, *, dynamic_batch: bool = True):
    generator = torch.Generator().manual_seed(17)
    batches = (1, 2) if dynamic_batch else (1,)
    return [
        torch.randn(batch, *example.shape[1:], generator=generator)
        for batch in batches
    ]


def assert_parity(reference, candidate, inputs, *, atol: float, rtol: float) -> None:
    with torch.inference_mode():
        for values in inputs:
            expected = reference(values).detach().cpu()
            actual = candidate(values).detach().cpu()
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
