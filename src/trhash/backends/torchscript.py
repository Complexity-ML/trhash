"""Framework-independent TorchScript backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..bundle import resolve_bundle
from ..metadata import ModelMetadata
from .portable import PortableDetectionBackend


class TorchScriptBackend(PortableDetectionBackend):
    def __init__(
        self,
        model: Union[str, Path],
        *,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("TorchScript inference requires PyTorch") from error
        self.torch = torch
        self.model_id = str(model)
        self.bundle = resolve_bundle(model, revision=revision, token=token)
        self.metadata = ModelMetadata.load(self.bundle)
        if device is not None:
            selected_device = device
        elif torch.cuda.is_available():
            selected_device = "cuda"
        elif torch.backends.mps.is_available():
            selected_device = "mps"
        else:
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        # Loading directly onto MPS can fail on harmless traced scalar constants.
        self.model = (
            torch.jit.load(str(self.bundle / self.metadata.model_file), map_location="cpu")
            .eval()
            .to(self.device)
        )
        self.names = self.metadata.class_names
        self.providers = [f"TorchScript:{self.device.type}"]

    def _predict_raw(self, pixels: np.ndarray):
        with self.torch.inference_mode():
            outputs = self.model(self.torch.from_numpy(pixels).to(self.device))
            if isinstance(outputs, (tuple, list)):
                return tuple(output.cpu().numpy() for output in outputs)
            return outputs.cpu().numpy()

    def serve(self, **options) -> None:
        from ..server.runner import run_server

        run_server(self.model_id, device=str(self.device), **options)
