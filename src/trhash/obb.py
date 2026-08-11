"""Oriented bounding-box result, transport, and rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from PIL import Image, ImageDraw


OrientedBox = tuple[float, float, float, float, float]


def _corners(box: OrientedBox) -> list[tuple[float, float]]:
    x1, y1, x2, y2, angle = box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = (x2 - x1) / 2.0
    half_height = (y2 - y1) / 2.0
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        (
            center_x + local_x * cosine - local_y * sine,
            center_y + local_x * sine + local_y * cosine,
        )
        for local_x, local_y in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    ]


@dataclass
class OBBResult:
    image: Image.Image = field(repr=False)
    boxes: list[OrientedBox]
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
        if not (len(self.boxes) == len(self.scores) == len(self.labels)):
            raise ValueError("boxes, scores, and labels must have equal lengths")
        self.boxes = [tuple(float(value) for value in box) for box in self.boxes]
        if any(len(box) != 5 for box in self.boxes):
            raise ValueError("oriented boxes require x1, y1, x2, y2, and angle")

    @classmethod
    def from_payload(cls, image: Image.Image, payload: Dict[str, Any]) -> "OBBResult":
        detections = payload.get("detections", [])
        largest_label = max((int(item["label"]) for item in detections), default=-1)
        names = [str(index) for index in range(largest_label + 1)]
        for item in detections:
            if "class_name" in item:
                names[int(item["label"])] = str(item["class_name"])
        return cls(
            image=image,
            boxes=[
                (
                    *(float(value) for value in item["box_xyxy"]),
                    float(item["angle_radians"]),
                )
                for item in detections
            ],
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
        for box, score, label in zip(self.boxes, self.scores, self.labels):
            detections.append(
                {
                    "box_xyxy": list(box[:4]),
                    "angle_radians": box[4],
                    "score": score,
                    "label": label,
                    "class_name": (
                        self.names[label] if label < len(self.names) else str(label)
                    ),
                }
            )
        payload = {
            "task": "obb",
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
        line_width: Optional[int] = None,
        *,
        labels: bool = True,
        conf: bool = True,
    ) -> Image.Image:
        rendered = self.image.copy()
        draw = ImageDraw.Draw(rendered)
        width = line_width or max(round(sum(rendered.size) * 0.0015), 2)
        for box, score, label in zip(self.boxes, self.scores, self.labels):
            color = (
                50 + (37 * label + 53) % 205,
                50 + (97 * label + 29) % 205,
                50 + (17 * label + 113) % 205,
            )
            corners = _corners(box)
            draw.line((*corners, corners[0]), fill=color, width=width, joint="curve")
            parts = []
            if labels:
                parts.append(self.names[label] if label < len(self.names) else str(label))
            if conf:
                parts.append(f"{score:.2f}")
            if parts:
                anchor = min(corners, key=lambda point: point[1])
                draw.text((anchor[0] + 2, max(0.0, anchor[1] - 12)), " ".join(parts), fill=color)
        return rendered

    def show(self, **plot_options) -> Image.Image:
        rendered = self.plot(**plot_options)
        rendered.show(title=self.source or "TR-Hash OBB")
        return rendered

    def save(self, path: Union[str, Path], **plot_options) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.plot(**plot_options).save(output)
        return output.resolve()
