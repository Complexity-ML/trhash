"""Thin public model facade."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
from typing import Optional, Union

from .backends.remote import RemoteBackend
from .classification import ClassificationResult
from .depth import DepthResult
from .pose import PoseResult
from .result import Result
from .segmentation import SemanticSegmentationResult
from .sources import (
    PredictionSource,
    attach_source_metadata,
    chunks,
    expand_sources,
    inference_source,
)

VisionResult = Union[
    Result,
    ClassificationResult,
    SemanticSegmentationResult,
    DepthResult,
    PoseResult,
]
PredictionOutput = Union[VisionResult, list[VisionResult], Iterator[VisionResult]]


class Vision:
    """Load a TR-Hash model once, then predict, fine-tune, or serve it."""

    def __init__(
        self,
        model: Union[str, Path],
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        runtime: str = "auto",
    ) -> None:
        if endpoint:
            self.backend = RemoteBackend(str(model), endpoint, api_key=api_key)
        else:
            path = Path(model).expanduser()
            use_portable = runtime in {"onnx", "torchscript", "coreml", "tensorrt"} or (
                runtime == "auto" and (not path.is_dir() or (path / "trhash.json").exists())
            )
            if use_portable:
                from .backends.portable import load_portable_backend

                self.backend = load_portable_backend(
                    model,
                    runtime=runtime,
                    device=device,
                    revision=revision,
                    token=token,
                )
            elif runtime in {"auto", "torch"}:
                from .backends.local import LocalBackend

                self.backend = LocalBackend(model, device=device, revision=revision, token=token)
            else:
                raise ValueError(
                    "runtime must be auto, onnx, torchscript, coreml, tensorrt, or torch"
                )

    def predict(
        self,
        source: PredictionSource,
        *,
        confidence: Optional[float] = None,
        iou: float = 0.45,
        batch: int = 1,
        stream: bool = False,
    ) -> PredictionOutput:
        sources, single = expand_sources(source)

        def generate() -> Iterator[VisionResult]:
            try:
                predict_batch = getattr(self.backend, "predict_batch", None)
                for group in chunks(sources, batch):
                    inference_group = [inference_source(item) for item in group]
                    if predict_batch is not None:
                        predicted = predict_batch(
                            inference_group,
                            confidence=confidence,
                            iou=iou,
                        )
                    else:
                        predicted = [
                            self.backend.predict(item, confidence=confidence, iou=iou)
                            for item in inference_group
                        ]
                    if len(predicted) != len(group):
                        raise RuntimeError("prediction backend returned the wrong batch size")
                    for item, result in zip(group, predicted):
                        attach_source_metadata(result, item)
                        yield result
            finally:
                close = getattr(sources, "close", None)
                if close is not None:
                    close()

        results = generate()
        if stream:
            return results
        if single:
            return next(results)
        return list(results)

    __call__ = predict

    def track(
        self,
        source: PredictionSource,
        *,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        new_track_threshold: Optional[float] = None,
        match_iou_threshold: float = 0.3,
        second_match_iou_threshold: float = 0.2,
        track_buffer: int = 30,
        iou: float = 0.45,
        batch: int = 1,
        stream: bool = False,
        persist: bool = False,
    ) -> Union[list[Result], Iterator[Result]]:
        try:
            from .tracking import ByteTracker
        except ImportError as error:
            raise RuntimeError(
                'tracking requires `pip install "trhash[tracking]"`'
            ) from error

        if persist and hasattr(self, "_tracker"):
            tracker = self._tracker
        else:
            tracker = ByteTracker(
                high_threshold=high_threshold,
                low_threshold=low_threshold,
                new_track_threshold=new_track_threshold,
                match_iou_threshold=match_iou_threshold,
                second_match_iou_threshold=second_match_iou_threshold,
                track_buffer=track_buffer,
            )
            if persist:
                self._tracker = tracker
        detections = self.predict(
            source,
            confidence=low_threshold,
            iou=iou,
            batch=batch,
            stream=True,
        )

        def generate() -> Iterator[Result]:
            try:
                for result in detections:
                    if not isinstance(result, Result):
                        raise RuntimeError("tracking requires a detection model")
                    result.track_ids = tracker.update(
                        result.boxes,
                        result.scores,
                        result.labels,
                    )
                    yield result
            finally:
                close = getattr(detections, "close", None)
                if close is not None:
                    close()

        tracked = generate()
        return tracked if stream else list(tracked)

    def train(self, **options) -> Path:
        train = getattr(self.backend, "train", None)
        if train is None:
            raise RuntimeError("remote fine-tuning requires a managed training endpoint")
        return train(**options)

    sft = train

    def val(self, **options):
        from .validation import validate

        return validate(self, **options)

    def export(self, **options) -> Path:
        export = getattr(self.backend, "export", None)
        if export is None:
            raise RuntimeError("export requires a local PyTorch checkpoint")
        return export(**options)

    def benchmark(self, source: PredictionSource, **options):
        from .benchmarking import benchmark_model

        return benchmark_model(self, source, **options)

    def serve(self, **options) -> None:
        serve = getattr(self.backend, "serve", None)
        if serve is None:
            raise RuntimeError("a remote model is already served by its endpoint")
        serve(**options)

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "Vision":
        return self

    def __exit__(self, *_errors) -> None:
        self.close()
