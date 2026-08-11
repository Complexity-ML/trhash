"""Native Core ML runtime backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from ..bundle import resolve_bundle
from ..metadata import ModelMetadata
from .portable import PortableDetectionBackend


def _compute_units(ct, requested: Optional[str]):
    choices = {
        None: ct.ComputeUnit.ALL,
        "auto": ct.ComputeUnit.ALL,
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }
    try:
        return choices[requested]
    except KeyError as error:
        raise ValueError(
            "CoreML device must be auto, all, cpu, cpu_and_gpu, or cpu_and_ne"
        ) from error


class CoreMLBackend(PortableDetectionBackend):
    def __init__(
        self,
        model: Union[str, Path],
        *,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        try:
            import coremltools as ct
        except ImportError as error:
            raise RuntimeError('CoreML inference requires `pip install "trhash[coreml]"`') from error
        self.model_id = str(model)
        self.bundle = resolve_bundle(model, revision=revision, token=token)
        self.metadata = ModelMetadata.load(self.bundle)
        self.compute_units = _compute_units(ct, device)
        self.model = ct.models.MLModel(
            str(self.bundle / self.metadata.model_file),
            compute_units=self.compute_units,
        )
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name
        self.names = self.metadata.class_names
        self.providers = [f"CoreML:{self.compute_units.name}"]

    def _predict_raw(self, pixels: np.ndarray) -> np.ndarray:
        return self.model.predict({self.input_name: pixels})[self.output_name]

    def predict_batch(self, sources: Sequence, **options):
        results = []
        for source in sources:
            results.extend(super().predict_batch((source,), **options))
        return results

    def serve(self, **options) -> None:
        from ..server.runner import run_server

        device = {
            "ALL": "all",
            "CPU_ONLY": "cpu",
            "CPU_AND_GPU": "cpu_and_gpu",
            "CPU_AND_NE": "cpu_and_ne",
        }[self.compute_units.name]
        run_server(self.model_id, device=device, **options)
