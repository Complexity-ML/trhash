"""Small HTTP-only inference backend."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional, Union

import httpx
from PIL import Image

from ..result import Result

ImageSource = Union[str, Path, Image.Image]


class RemoteBackend:
    def __init__(
        self,
        model: str,
        endpoint: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.model_id = model
        self.endpoint = endpoint.rstrip("/")
        headers = {"X-API-Key": api_key} if api_key else None
        self.client = httpx.Client(headers=headers, timeout=timeout)

    @staticmethod
    def _image_bytes(source: ImageSource) -> tuple[Image.Image, bytes, str]:
        if isinstance(source, Image.Image):
            image = source.copy().convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            return image, buffer.getvalue(), "image.jpg"
        path = Path(source).expanduser()
        return Image.open(path).convert("RGB"), path.read_bytes(), path.name

    def predict(
        self,
        source: ImageSource,
        *,
        confidence: Optional[float] = None,
        iou: float = 0.45,
    ) -> Result:
        started = time.perf_counter()
        image, content, filename = self._image_bytes(source)
        params = {"iou_threshold": iou}
        if confidence is not None:
            params["confidence"] = confidence
        response = self.client.post(
            f"{self.endpoint}/v1/predict",
            params=params,
            files={"file": (filename, content, "application/octet-stream")},
        )
        response.raise_for_status()
        result = Result.from_payload(image, response.json())
        result.speed["network"] = (time.perf_counter() - started) * 1000.0
        result.source = None if isinstance(source, Image.Image) else str(source)
        return result

    def class_names(self) -> tuple[str, ...]:
        response = self.client.get(f"{self.endpoint}/v1/model")
        response.raise_for_status()
        payload = response.json()
        metadata = payload.get("metadata", {})
        names = metadata.get("class_names", payload.get("class_names"))
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError("remote endpoint does not expose model class_names")
        return tuple(str(name) for name in names)

    def close(self) -> None:
        self.client.close()
