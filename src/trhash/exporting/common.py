"""Shared preparation and parity checks for portable exports."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Union

import torch


class ExportVisionModel(torch.nn.Module):
    def __init__(self, model, task: str):
        super().__init__()
        self.model = model
        self.task = task
        self.output_names = (
            ("logits",)
            if task in {"classification", "semantic_segmentation"}
            else ("predictions",)
        )

    def forward(self, pixel_values):
        if self.task == "classification":
            return self.model(pixel_values)["logits"]
        if self.task == "semantic_segmentation":
            return self.model(pixel_values)["logits"]
        if self.task == "detection":
            return self.model.forward_predictions(pixel_values)
        raise NotImplementedError(f"portable export is not implemented for task={self.task}")


def prepare_export(backend, output: Union[str, Path]):
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    task = str(getattr(backend, "task", getattr(backend.model, "vision_task", "detection")))
    detector = ExportVisionModel(copy.deepcopy(backend.model).to("cpu").eval(), task)
    config = getattr(backend.model, "config", None) or getattr(
        backend.model, "detector_config", None
    )
    size = config.image_size
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
