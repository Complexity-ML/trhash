"""Shared detection pipeline and runtime selection for portable bundles."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image

from ..bundle import resolve_bundle
from ..decoding import decode
from ..metadata import ModelMetadata
from ..preprocessing import preprocess, restore_boxes
from ..result import Result

ImageSource = Union[str, Path, Image.Image]


class PortableDetectionBackend:
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
        started = time.perf_counter()
        images = [
            (source.copy() if isinstance(source, Image.Image) else Image.open(source)).convert("RGB")
            for source in sources
        ]
        prepared = [preprocess(image, self.metadata) for image in images]
        pixels = np.stack([item[0] for item in prepared])
        preprocessed = time.perf_counter()
        predictions = self._predict_raw(pixels)
        inferred = time.perf_counter()
        threshold = (
            float(confidence)
            if confidence is not None
            else self.metadata.recommended_confidence
        )
        results = []
        for source, image, raw, (_, geometry) in zip(sources, images, predictions, prepared):
            boxes, scores, labels = decode(raw, self.metadata, confidence=threshold, iou=iou)
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
        finished = time.perf_counter()
        count = max(len(images), 1)
        speed = {
            "preprocess": (preprocessed - started) * 1000.0 / count,
            "inference": (inferred - preprocessed) * 1000.0 / count,
            "postprocess": (finished - inferred) * 1000.0 / count,
        }
        for result in results:
            result.speed.update(speed)
        return results


def load_portable_backend(
    model,
    *,
    runtime: str = "auto",
    device: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
):
    bundle = resolve_bundle(model, revision=revision, token=token)
    metadata = ModelMetadata.load(bundle)
    extension = Path(metadata.model_file).suffix.casefold()
    detected_runtime = "onnx" if extension == ".onnx" else "torchscript"
    if extension not in {".onnx", ".torchscript"}:
        raise ValueError(f"unsupported portable model file: {metadata.model_file}")
    if runtime != "auto" and runtime != detected_runtime:
        raise ValueError(
            f"bundle contains {detected_runtime}, but runtime={runtime} was requested"
        )
    if detected_runtime == "onnx":
        from .onnx import OnnxBackend

        return OnnxBackend(bundle, device=device)
    from .torchscript import TorchScriptBackend

    return TorchScriptBackend(bundle, device=device)
