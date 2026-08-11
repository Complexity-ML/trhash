"""Thin public model facade."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .backends.remote import RemoteBackend
from .result import Result

ImageSource = Union[str, Path, Image.Image]


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
            use_onnx = runtime == "onnx" or (runtime == "auto" and (not path.is_dir() or (path / "trhash.json").exists()))
            if use_onnx:
                from .backends.onnx import OnnxBackend

                self.backend = OnnxBackend(model, device=device, revision=revision, token=token)
            elif runtime in {"auto", "torch"}:
                from .backends.local import LocalBackend

                self.backend = LocalBackend(model, device=device, revision=revision, token=token)
            else:
                raise ValueError("runtime must be auto, onnx, or torch")

    def predict(
        self,
        source: ImageSource,
        *,
        confidence: Optional[float] = None,
        iou: float = 0.45,
    ) -> Result:
        return self.backend.predict(source, confidence=confidence, iou=iou)

    __call__ = predict

    def train(self, **options) -> Path:
        train = getattr(self.backend, "train", None)
        if train is None:
            raise RuntimeError("remote fine-tuning requires a managed training endpoint")
        return train(**options)

    sft = train

    def export(self, **options) -> Path:
        export = getattr(self.backend, "export", None)
        if export is None:
            raise RuntimeError("export requires a local PyTorch checkpoint")
        return export(**options)

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
