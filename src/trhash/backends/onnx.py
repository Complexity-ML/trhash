"""Framework-independent ONNX Runtime backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..bundle import resolve_bundle
from ..metadata import ModelMetadata
from .portable import PortableDetectionBackend


def _providers(ort, requested: Optional[str]) -> list[str]:
    available = set(ort.get_available_providers())
    choices = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
    }
    if requested and requested != "auto":
        provider = choices.get(requested, requested)
        if provider not in available:
            raise RuntimeError(f"ONNX Runtime provider is unavailable: {provider}")
        return [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider]
    return [
        provider
        for provider in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]


class OnnxBackend(PortableDetectionBackend):
    def __init__(
        self,
        model: Union[str, Path],
        *,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError('ONNX inference requires `pip install "trhash[runtime]"`') from error
        self.model_id = str(model)
        self.bundle = resolve_bundle(model, revision=revision, token=token)
        self.metadata = ModelMetadata.load(self.bundle)
        self.providers = _providers(ort, device)
        self.session = ort.InferenceSession(
            str(self.bundle / self.metadata.model_file),
            providers=self.providers,
        )
        self.names = self.metadata.class_names

    def _predict_raw(self, pixels: np.ndarray):
        outputs = self.session.run(
            self.metadata.output_names,
            {"pixel_values": pixels},
        )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def serve(self, **options) -> None:
        from ..server.runner import run_server

        run_server(self.model_id, device=self.providers[0], **options)
