"""Metric depth result, lossless transport, and color rendering."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image, ImageDraw


def _pixels(image: Image.Image):
    flattened = getattr(image, "get_flattened_data", None)
    return flattened() if flattened is not None else image.getdata()


def _depth_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    red = max(0.0, min(1.0, 1.5 - abs(4.0 * value - 3.0)))
    green = max(0.0, min(1.0, 1.5 - abs(4.0 * value - 2.0)))
    blue = max(0.0, min(1.0, 1.5 - abs(4.0 * value - 1.0)))
    return round(red * 255), round(green * 255), round(blue * 255)


@dataclass
class DepthResult:
    image: Image.Image = field(repr=False)
    depth: Image.Image = field(repr=False)
    source: Optional[str] = None
    speed: Dict[str, float] = field(default_factory=dict)
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    fps: Optional[float] = None

    def __post_init__(self) -> None:
        self.image = self.image.copy().convert("RGB")
        self.depth = self.depth.copy().convert("F")
        if self.depth.size != self.image.size:
            raise ValueError("depth map size must match the source image")
        if not any(math.isfinite(float(value)) for value in _pixels(self.depth)):
            raise ValueError("depth map must contain finite values")

    @property
    def min_depth(self) -> float:
        return min(
            float(value)
            for value in _pixels(self.depth)
            if math.isfinite(float(value))
        )

    @property
    def max_depth(self) -> float:
        return max(
            float(value)
            for value in _pixels(self.depth)
            if math.isfinite(float(value))
        )

    @classmethod
    def from_payload(cls, image: Image.Image, payload: Dict[str, Any]) -> "DepthResult":
        encoded = payload.get("depth", {})
        if encoded.get("encoding") != "tiff_float32_base64" or not encoded.get("data"):
            raise ValueError("depth response is missing a float32 TIFF map")
        depth = Image.open(io.BytesIO(base64.b64decode(encoded["data"]))).copy()
        return cls(
            image=image,
            depth=depth,
            source=payload.get("source"),
            speed={
                str(name): float(value)
                for name, value in payload.get("speed", {}).items()
            },
            frame_index=payload.get("frame_index"),
            timestamp=payload.get("timestamp"),
            fps=payload.get("fps"),
        )

    def to_dict(self) -> Dict[str, Any]:
        buffer = io.BytesIO()
        self.depth.save(buffer, format="TIFF", compression="tiff_deflate")
        payload = {
            "task": "depth",
            "image": {"width": self.image.width, "height": self.image.height},
            "depth": {
                "encoding": "tiff_float32_base64",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "min": self.min_depth,
                "max": self.max_depth,
            },
        }
        if self.source is not None:
            payload["source"] = self.source
        if self.speed:
            payload["speed"] = dict(self.speed)
        if self.frame_index is not None:
            payload["frame_index"] = self.frame_index
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        if self.fps is not None:
            payload["fps"] = self.fps
        return payload

    def plot(
        self,
        *,
        alpha: float = 1.0,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        legend: bool = True,
    ) -> Image.Image:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        lower = self.min_depth if min_depth is None else float(min_depth)
        upper = self.max_depth if max_depth is None else float(max_depth)
        if upper <= lower:
            upper = lower + 1e-6
        scale = upper - lower
        colors = []
        for raw in _pixels(self.depth):
            value = float(raw)
            normalized = 0.0 if not math.isfinite(value) else (value - lower) / scale
            colors.append(_depth_color(normalized))
        colored = Image.new("RGB", self.depth.size)
        colored.putdata(colors)
        rendered = Image.blend(self.image, colored, alpha)
        if legend:
            draw = ImageDraw.Draw(rendered)
            text = f"{lower:.2f} – {upper:.2f}"
            width = draw.textlength(text) + 12
            draw.rectangle((4, 4, width + 4, 22), fill=(0, 0, 0))
            draw.text((10, 7), text, fill=(255, 255, 255))
        return rendered

    def show(self, **plot_options) -> Image.Image:
        rendered = self.plot(**plot_options)
        rendered.show(title=self.source or "TR-Hash depth")
        return rendered

    def save(self, path: Union[str, Path], **plot_options) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.plot(**plot_options).save(output)
        return output.resolve()

    def save_depth(self, path: Union[str, Path]) -> Path:
        output = Path(path).expanduser()
        if output.suffix.casefold() not in {".tif", ".tiff"}:
            raise ValueError("raw float32 depth maps must use a .tif or .tiff path")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.depth.save(output, format="TIFF", compression="tiff_deflate")
        return output.resolve()
