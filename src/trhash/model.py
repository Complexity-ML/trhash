"""Thin public model facade."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
from typing import Optional, Union

from .backends.remote import RemoteBackend
from .result import Result
from .sources import PredictionSource, chunks, expand_sources

PredictionOutput = Union[Result, list[Result], Iterator[Result]]


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
            use_portable = runtime in {"onnx", "torchscript"} or (
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
                raise ValueError("runtime must be auto, onnx, torchscript, or torch")

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

        def generate() -> Iterator[Result]:
            predict_batch = getattr(self.backend, "predict_batch", None)
            for group in chunks(sources, batch):
                if predict_batch is not None:
                    yield from predict_batch(group, confidence=confidence, iou=iou)
                else:
                    for item in group:
                        yield self.backend.predict(item, confidence=confidence, iou=iou)

        results = generate()
        if stream:
            return results
        if single:
            return next(results)
        return list(results)

    __call__ = predict

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
