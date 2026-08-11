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
    track_ids: Optional[List[Optional[int]]] = None
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    fps: Optional[float] = None

    def __post_init__(self) -> None:
        if not (len(self.boxes) == len(self.scores) == len(self.labels)):
            raise ValueError("boxes, scores, and labels must have equal lengths")
        if self.track_ids is not None and len(self.track_ids) != len(self.boxes):
            raise ValueError("track_ids must align with boxes")

    def _aligned_track_ids(self) -> List[Optional[int]]:
        return [None] * len(self.boxes) if self.track_ids is None else self.track_ids

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
            track_ids=(
                [
                    int(item["track_id"]) if item.get("track_id") is not None else None
                    for item in detections
                ]
                if any("track_id" in item for item in detections)
                else None
            ),
            source=payload.get("source"),
            frame_index=payload.get("frame_index"),
            timestamp=payload.get("timestamp"),
            fps=payload.get("fps"),
            speed={
                str(name): float(value)
                for name, value in payload.get("speed", {}).items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        detections = []
        track_ids = self._aligned_track_ids()
        for box, score, label, track_id in zip(
            self.boxes,
            self.scores,
            self.labels,
            track_ids,
        ):
            name = self.names[label] if label < len(self.names) else str(label)
            detection = {
                "box_xyxy": list(box),
                "score": score,
                "label": label,
                "class_name": name,
            }
            if track_id is not None:
                detection["track_id"] = track_id
            detections.append(detection)
        payload = {
            "task": "detection",
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
        width = line_width or max(round((rendered.width + rendered.height) * 0.0015), 2)
        track_ids = self._aligned_track_ids()
        for box, score, label, track_id in zip(
            self.boxes,
            self.scores,
            self.labels,
            track_ids,
        ):
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
                parts.append(f"{name} #{track_id}" if track_id is not None else name)
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


def result_from_payload(image: Image.Image, payload: Dict[str, Any]):
    task = payload.get("task", "detection")
    if task == "classification":
        from .classification import ClassificationResult

        return ClassificationResult.from_payload(image, payload)
    if task == "detection":
        return Result.from_payload(image, payload)
    raise ValueError(f"unsupported prediction task: {task}")
