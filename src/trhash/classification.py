"""Classification prediction result and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image, ImageDraw


@dataclass
class ClassificationResult:
    image: Image.Image = field(repr=False)
    scores: List[float]
    labels: List[int]
    names: Sequence[str]
    source: Optional[str] = None
    speed: Dict[str, float] = field(default_factory=dict)
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    fps: Optional[float] = None

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.labels):
            raise ValueError("classification scores and labels must align")

    @property
    def top1(self) -> int:
        if not self.labels:
            raise ValueError("classification result is empty")
        return self.labels[0]

    @property
    def top1_confidence(self) -> float:
        if not self.scores:
            raise ValueError("classification result is empty")
        return self.scores[0]

    @classmethod
    def from_payload(
        cls,
        image: Image.Image,
        payload: Dict[str, Any],
    ) -> "ClassificationResult":
        predictions = payload.get("predictions", [])
        largest = max((int(item["label"]) for item in predictions), default=-1)
        names = [str(index) for index in range(largest + 1)]
        for item in predictions:
            names[int(item["label"])] = str(item.get("class_name", item["label"]))
        return cls(
            image=image.copy().convert("RGB"),
            scores=[float(item["score"]) for item in predictions],
            labels=[int(item["label"]) for item in predictions],
            names=tuple(names),
            source=payload.get("source"),
            frame_index=payload.get("frame_index"),
            timestamp=payload.get("timestamp"),
            fps=payload.get("fps"),
            speed={
                str(name): float(value)
                for name, value in payload.get("speed", {}).items()
            },
        )

    def to_dict(self, *, top_k: Optional[int] = None) -> Dict[str, Any]:
        limit = len(self.labels) if top_k is None else max(0, min(top_k, len(self.labels)))
        predictions = []
        for label, score in zip(self.labels[:limit], self.scores[:limit]):
            name = self.names[label] if label < len(self.names) else str(label)
            predictions.append(
                {"label": label, "class_name": name, "score": score}
            )
        payload = {
            "task": "classification",
            "image": {"width": self.image.width, "height": self.image.height},
            "predictions": predictions,
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

    def plot(self, *, top_k: int = 5) -> Image.Image:
        rendered = self.image.copy().convert("RGB")
        draw = ImageDraw.Draw(rendered)
        lines = []
        for label, score in zip(self.labels[:top_k], self.scores[:top_k]):
            name = self.names[label] if label < len(self.names) else str(label)
            lines.append(f"{name} {score:.3f}")
        if lines:
            line_height = 14
            width = max(draw.textlength(line) for line in lines) + 12
            height = len(lines) * line_height + 8
            draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
            for index, line in enumerate(lines):
                draw.text((6, 4 + index * line_height), line, fill=(255, 255, 255))
        return rendered

    def show(self, **plot_options) -> Image.Image:
        rendered = self.plot(**plot_options)
        rendered.show(title=self.source or "TR-Hash classification")
        return rendered

    def save(self, path: Union[str, Path], **plot_options) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.plot(**plot_options).save(output)
        return output.resolve()
