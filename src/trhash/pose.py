"""Pose keypoint result, transport, and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from PIL import Image, ImageDraw


Keypoint = tuple[float, float, float]


@dataclass
class PoseResult:
    image: Image.Image = field(repr=False)
    keypoints: list[Keypoint]
    names: Sequence[str]
    source: Optional[str] = None
    speed: Dict[str, float] = field(default_factory=dict)
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    fps: Optional[float] = None

    def __post_init__(self) -> None:
        self.image = self.image.copy().convert("RGB")
        if len(self.keypoints) != len(self.names):
            raise ValueError("keypoints and names must have equal lengths")
        self.keypoints = [
            (float(x), float(y), float(score)) for x, y, score in self.keypoints
        ]

    @classmethod
    def from_payload(cls, image: Image.Image, payload: Dict[str, Any]) -> "PoseResult":
        values = sorted(payload.get("keypoints", []), key=lambda item: int(item["index"]))
        return cls(
            image=image,
            keypoints=[
                (float(item["x"]), float(item["y"]), float(item["score"]))
                for item in values
            ],
            names=tuple(str(item.get("name", item["index"])) for item in values),
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
        payload = {
            "task": "pose",
            "image": {"width": self.image.width, "height": self.image.height},
            "keypoints": [
                {
                    "index": index,
                    "name": self.names[index],
                    "x": x,
                    "y": y,
                    "score": score,
                }
                for index, (x, y, score) in enumerate(self.keypoints)
            ],
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
        confidence: float = 0.25,
        radius: Optional[int] = None,
        labels: bool = True,
        conf: bool = False,
        connections: Sequence[tuple[int, int]] = (),
    ) -> Image.Image:
        rendered = self.image.copy()
        draw = ImageDraw.Draw(rendered)
        point_radius = radius or max(round((rendered.width + rendered.height) * 0.003), 2)
        visible = [score >= confidence for _, _, score in self.keypoints]
        for start, end in connections:
            if not 0 <= start < len(self.keypoints) or not 0 <= end < len(self.keypoints):
                raise ValueError("pose connection index is out of range")
            if visible[start] and visible[end]:
                draw.line(
                    (self.keypoints[start][:2], self.keypoints[end][:2]),
                    fill=(80, 220, 120),
                    width=max(point_radius, 2),
                )
        for index, ((x, y, score), is_visible) in enumerate(
            zip(self.keypoints, visible)
        ):
            if not is_visible:
                continue
            color = (
                50 + (37 * index + 53) % 205,
                50 + (97 * index + 29) % 205,
                50 + (17 * index + 113) % 205,
            )
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=color,
                outline=(0, 0, 0),
            )
            parts = []
            if labels:
                parts.append(str(self.names[index]))
            if conf:
                parts.append(f"{score:.2f}")
            if parts:
                draw.text((x + point_radius + 2, y - point_radius), " ".join(parts), fill=color)
        return rendered

    def show(self, **plot_options) -> Image.Image:
        rendered = self.plot(**plot_options)
        rendered.show(title=self.source or "TR-Hash pose")
        return rendered

    def save(self, path: Union[str, Path], **plot_options) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.plot(**plot_options).save(output)
        return output.resolve()
