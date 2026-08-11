"""Semantic segmentation result, mask serialization, and rendering."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from PIL import Image, ImageDraw


def _class_color(label: int) -> tuple[int, int, int]:
    return (
        50 + (37 * label + 53) % 205,
        50 + (97 * label + 29) % 205,
        50 + (17 * label + 113) % 205,
    )


def _pixels(image: Image.Image):
    flattened = getattr(image, "get_flattened_data", None)
    return flattened() if flattened is not None else image.getdata()


@dataclass
class SemanticSegmentationResult:
    image: Image.Image = field(repr=False)
    mask: Image.Image = field(repr=False)
    names: Sequence[str]
    source: Optional[str] = None
    speed: Dict[str, float] = field(default_factory=dict)
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    fps: Optional[float] = None

    def __post_init__(self) -> None:
        self.image = self.image.copy().convert("RGB")
        if self.mask.size != self.image.size:
            raise ValueError("semantic mask size must match the source image")
        if self.mask.mode not in {"L", "I", "I;16"}:
            self.mask = self.mask.convert("I")
        _, largest = self.mask.getextrema()
        if largest <= 255:
            self.mask = self.mask.convert("L")
        elif largest <= 65535:
            self.mask = self.mask.convert("I;16")
        else:
            raise ValueError("semantic masks support at most 65536 classes")

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(sorted(set(int(value) for value in _pixels(self.mask))))

    def _segments(self) -> list[Dict[str, Any]]:
        counts: Dict[int, int] = {}
        for value in _pixels(self.mask):
            label = int(value)
            counts[label] = counts.get(label, 0) + 1
        total = max(self.mask.width * self.mask.height, 1)
        return [
            {
                "label": label,
                "class_name": self.names[label] if label < len(self.names) else str(label),
                "pixels": count,
                "fraction": count / total,
            }
            for label, count in sorted(counts.items())
        ]

    @classmethod
    def from_payload(
        cls,
        image: Image.Image,
        payload: Dict[str, Any],
    ) -> "SemanticSegmentationResult":
        encoded = payload.get("mask", {})
        if encoded.get("encoding") != "png_base64" or not encoded.get("data"):
            raise ValueError("semantic response is missing a PNG mask")
        mask = Image.open(io.BytesIO(base64.b64decode(encoded["data"]))).copy()
        segments = payload.get("segments", [])
        largest = max((int(item["label"]) for item in segments), default=-1)
        names = [str(index) for index in range(largest + 1)]
        for item in segments:
            names[int(item["label"])] = str(item.get("class_name", item["label"]))
        return cls(
            image=image,
            mask=mask,
            names=tuple(names),
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
        self.mask.save(buffer, format="PNG")
        payload = {
            "task": "semantic_segmentation",
            "image": {"width": self.image.width, "height": self.image.height},
            "mask": {
                "encoding": "png_base64",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            },
            "segments": self._segments(),
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

    def plot(self, *, alpha: float = 0.45, legend: bool = True) -> Image.Image:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        colored = Image.new("RGB", self.mask.size)
        colored.putdata([_class_color(int(label)) for label in _pixels(self.mask)])
        rendered = Image.blend(self.image, colored, alpha)
        if legend:
            draw = ImageDraw.Draw(rendered)
            y = 4
            for label in self.labels:
                name = self.names[label] if label < len(self.names) else str(label)
                color = _class_color(label)
                text_width = draw.textlength(name)
                draw.rectangle((4, y, text_width + 16, y + 14), fill=color)
                draw.text((10, y + 1), name, fill=(0, 0, 0))
                y += 17
        return rendered

    def show(self, **plot_options) -> Image.Image:
        rendered = self.plot(**plot_options)
        rendered.show(title=self.source or "TR-Hash semantic segmentation")
        return rendered

    def save(self, path: Union[str, Path], **plot_options) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.plot(**plot_options).save(output)
        return output.resolve()

    def save_mask(self, path: Union[str, Path]) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.mask.save(output)
        return output.resolve()
