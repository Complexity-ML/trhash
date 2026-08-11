"""Small HTTP-only inference backend."""

from __future__ import annotations

import io
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
        return Result.from_payload(image, response.json())

    def close(self) -> None:
        self.client.close()
