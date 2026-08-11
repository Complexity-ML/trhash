"""Framework-independent ONNX Runtime backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from PIL import Image

from ..bundle import resolve_bundle
from ..decoding import decode
from ..metadata import ModelMetadata
from ..preprocessing import preprocess, restore_boxes
from ..result import Result

ImageSource = Union[str, Path, Image.Image]


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


class OnnxBackend:
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

    def predict(
        self,
        source: ImageSource,
        *,
        confidence: Optional[float] = None,
        iou: float = 0.45,
    ) -> Result:
        return self.predict_batch((source,), confidence=confidence, iou=iou)[0]

    def predict_batch(
        self,
        sources: Sequence[ImageSource],
        *,
        confidence: Optional[float] = None,
        iou: float = 0.45,
    ) -> list[Result]:
        images = [
            (source.copy() if isinstance(source, Image.Image) else Image.open(source)).convert("RGB")
            for source in sources
        ]
        prepared = [preprocess(image, self.metadata) for image in images]
        pixels = np.stack([item[0] for item in prepared])
        predictions = self.session.run(("predictions",), {"pixel_values": pixels})[0]
        threshold = (
            float(confidence)
            if confidence is not None
            else self.metadata.recommended_confidence
        )
        results = []
        for source, image, raw, (_, geometry) in zip(sources, images, predictions, prepared):
            boxes, scores, labels = decode(
                raw,
                self.metadata,
                confidence=threshold,
                iou=iou,
            )
            boxes = restore_boxes(boxes, self.metadata, geometry)
            results.append(
                Result(
                    image=image,
                    boxes=[tuple(float(value) for value in box) for box in boxes],
                    scores=[float(value) for value in scores],
                    labels=[int(value) for value in labels],
                    names=self.names,
                    source=None if isinstance(source, Image.Image) else str(source),
                )
            )
        return results

    def serve(self, **options) -> None:
        from ..server.runner import run_server

        run_server(self.model_id, device=self.providers[0], **options)
