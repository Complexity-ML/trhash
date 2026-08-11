"""Instance segmentation result, mask transport, and rendering."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from PIL import Image, ImageDraw


def _encode_mask(mask: Image.Image) -> str:
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_mask(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("L")


@dataclass
class InstanceSegmentationResult:
    image: Image.Image = field(repr=False)
    boxes: list[tuple[float, float, float, float]]
    masks: list[Image.Image] = field(repr=False)
    scores: list[float]
    labels: list[int]
    names: Sequence[str]
    source: Optional[str] = None
    speed: Dict[str, float] = field(default_factory=dict)
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    fps: Optional[float] = None

    def __post_init__(self) -> None:
        self.image = self.image.copy().convert("RGB")
        if not (
            len(self.boxes)
            == len(self.masks)
            == len(self.scores)
            == len(self.labels)
        ):
            raise ValueError("boxes, masks, scores, and labels must have equal lengths")
        normalized_masks = []
        for mask in self.masks:
            if mask.size != self.image.size:
                raise ValueError("instance mask size must match the source image")
            normalized_masks.append(mask.copy().convert("L"))
        self.masks = normalized_masks

    @classmethod
    def from_payload(
        cls,
        image: Image.Image,
        payload: Dict[str, Any],
    ) -> "InstanceSegmentationResult":
        detections = payload.get("detections", [])
        largest_label = max((int(item["label"]) for item in detections), default=-1)
        names = [str(index) for index in range(largest_label + 1)]
        for item in detections:
            if "class_name" in item:
                names[int(item["label"])] = str(item["class_name"])
        return cls(
            image=image,
            boxes=[
                tuple(float(value) for value in item["box_xyxy"])
                for item in detections
            ],
            masks=[_decode_mask(item["mask"]["data"]) for item in detections],
            scores=[float(item["score"]) for item in detections],
            labels=[int(item["label"]) for item in detections],
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
        detections = []
        for box, mask, score, label in zip(
            self.boxes,
            self.masks,
            self.scores,
            self.labels,
        ):
            detections.append(
                {
                    "box_xyxy": list(box),
                    "mask": {"encoding": "png_base64", "data": _encode_mask(mask)},
                    "score": score,
                    "label": label,
                    "class_name": (
                        self.names[label] if label < len(self.names) else str(label)
                    ),
                }
            )
        payload = {
            "task": "instance_segmentation",
            "image": {"width": self.image.width, "height": self.image.height},
            "detections": detections,
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
        alpha: float = 0.45,
        line_width: Optional[int] = None,
        labels: bool = True,
        conf: bool = True,
    ) -> Image.Image:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        rendered = self.image.copy()
        width = line_width or max(round(sum(rendered.size) * 0.0015), 2)
        for mask, label in zip(self.masks, self.labels):
            color = (
                50 + (37 * label + 53) % 205,
                50 + (97 * label + 29) % 205,
                50 + (17 * label + 113) % 205,
            )
            overlay = Image.new("RGB", rendered.size, color)
            opacity = mask.point(lambda value: round(value * alpha))
            rendered.paste(overlay, mask=opacity)
        draw = ImageDraw.Draw(rendered)
        for box, score, label in zip(self.boxes, self.scores, self.labels):
            color = (
                50 + (37 * label + 53) % 205,
                50 + (97 * label + 29) % 205,
                50 + (17 * label + 113) % 205,
            )
            draw.rectangle(box, outline=color, width=width)
            parts = []
            if labels:
                parts.append(self.names[label] if label < len(self.names) else str(label))
            if conf:
                parts.append(f"{score:.2f}")
            if parts:
                draw.text((box[0] + 2, max(0.0, box[1] - 12)), " ".join(parts), fill=color)
        return rendered

    def show(self, **plot_options) -> Image.Image:
        rendered = self.plot(**plot_options)
        rendered.show(title=self.source or "TR-Hash instance segmentation")
        return rendered

    def save(self, path: Union[str, Path], **plot_options) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.plot(**plot_options).save(output)
        return output.resolve()
