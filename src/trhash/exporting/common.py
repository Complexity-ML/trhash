"""Shared preparation and parity checks for portable exports."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Union

import torch


def tensor_outputs(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


class ExportVisionModel(torch.nn.Module):
    def __init__(self, model, task: str):
        super().__init__()
        self.model = model
        self.task = task
        self.output_names = {
            "classification": ("logits",),
            "semantic_segmentation": ("logits",),
            "depth": ("depth",),
            "pose": ("heatmaps",),
            "instance_segmentation": (
                "predictions",
                "mask_coefficients",
                "prototypes",
            ),
            "detection": ("predictions",),
        }.get(task, ("predictions",))

    def forward(self, pixel_values):
        if self.task == "classification":
            return self.model(pixel_values)["logits"]
        if self.task == "semantic_segmentation":
            return self.model(pixel_values)["logits"]
        if self.task == "depth":
            return self.model(pixel_values)["depth"]
        if self.task == "pose":
            return self.model(pixel_values)["heatmaps"]
        if self.task == "instance_segmentation":
            outputs = self.model.forward_instance(pixel_values)
            return (
                outputs["raw"],
                outputs["mask_coefficients"],
                outputs["prototypes"],
            )
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
            expected = tensor_outputs(reference(values))
            actual = tensor_outputs(candidate(values))
            if len(actual) != len(expected):
                raise AssertionError("exported model returned the wrong number of outputs")
            for actual_tensor, expected_tensor in zip(actual, expected):
                torch.testing.assert_close(
                    actual_tensor.detach().cpu(),
                    expected_tensor.detach().cpu(),
                    atol=atol,
                    rtol=rtol,
                )
