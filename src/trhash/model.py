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
    ) -> None:
        if endpoint:
            self.backend = RemoteBackend(str(model), endpoint, api_key=api_key)
        else:
            from .backends.local import LocalBackend

            self.backend = LocalBackend(
                model,
                device=device,
                revision=revision,
                token=token,
            )

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
