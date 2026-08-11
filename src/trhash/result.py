"""Prediction result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw


@dataclass
class Result:
    image: Image.Image = field(repr=False)
    boxes: List[Tuple[float, float, float, float]]
    scores: List[float]
    labels: List[int]
    names: Sequence[str]
    source: Optional[str] = None
    speed: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, image: Image.Image, payload: Dict[str, Any]) -> "Result":
        detections = payload.get("detections", [])
        largest_label = max((int(item["label"]) for item in detections), default=-1)
        names = [str(index) for index in range(largest_label + 1)]
        for item in detections:
            if "class_name" in item:
                names[int(item["label"])] = str(item["class_name"])
        return cls(
            image=image.copy().convert("RGB"),
            boxes=[tuple(float(value) for value in item["box_xyxy"]) for item in detections],
            scores=[float(item["score"]) for item in detections],
            labels=[int(item["label"]) for item in detections],
            names=tuple(names),
            speed={
                str(name): float(value)
                for name, value in payload.get("speed", {}).items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        detections = []
        for box, score, label in zip(self.boxes, self.scores, self.labels):
            name = self.names[label] if label < len(self.names) else str(label)
            detections.append(
                {
                    "box_xyxy": list(box),
                    "score": score,
                    "label": label,
                    "class_name": name,
                }
            )
        payload = {
            "image": {"width": self.image.width, "height": self.image.height},
            "detections": detections,
        }
        if self.source is not None:
            payload["source"] = self.source
        if self.speed:
            payload["speed"] = dict(self.speed)
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
        width = line_width or max(round((rendered.width + rendered.height) * 0.0015), 2)
        for box, score, label in zip(self.boxes, self.scores, self.labels):
            color = (
                50 + (37 * label + 53) % 205,
                50 + (97 * label + 29) % 205,
                50 + (17 * label + 113) % 205,
            )
            draw.rectangle(box, outline=color, width=width)
            if not labels and not conf:
                continue
            name = self.names[label] if label < len(self.names) else str(label)
            parts = []
            if labels:
                parts.append(name)
            if conf:
                parts.append(f"{score:.2f}")
            text = " ".join(parts)
            text_box = draw.textbbox((0, 0), text)
            text_height = text_box[3] - text_box[1]
            x, y = box[0], max(0.0, box[1] - text_height - 4)
            draw.rectangle((x, y, x + text_box[2] + 4, y + text_height + 4), fill=color)
            draw.text((x + 2, y + 2), text, fill=(0, 0, 0))
        return rendered

    def show(self, **plot_options) -> Image.Image:
        rendered = self.plot(**plot_options)
        rendered.show(title=self.source or "TR-Hash prediction")
        return rendered

    def save(self, path: Union[str, Path], **plot_options) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.plot(**plot_options).save(output)
        return output.resolve()
