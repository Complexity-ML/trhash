"""Optional PyTorch backend loaded only for local execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from ..result import Result
from ..runtime import imports, resolve_checkpoint, resolve_device

ImageSource = Union[str, Path, Image.Image]


class LocalBackend:
    def __init__(
        self,
        model: Union[str, Path],
        *,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        (
            self.torch,
            load_checkpoint,
            self.preprocess,
            self.restore_boxes,
        ) = imports()
        self.checkpoint = resolve_checkpoint(
            model,
            revision=revision,
            token=token,
        )
        self.model_id = str(model)
        self.device = resolve_device(self.torch, device)
        self.model = load_checkpoint(self.checkpoint, device=self.device)
        names_path = self.checkpoint / "class_names.json"
        self.names = (
            tuple(str(name) for name in json.loads(names_path.read_text()))
            if names_path.exists()
            else tuple(str(index) for index in range(self.model.config.num_classes))
        )
        if len(self.names) != self.model.config.num_classes:
            raise ValueError("class_names.json does not match the detector class count")
        validation_path = self.checkpoint / "validation.json"
        self.validation = (
            json.loads(validation_path.read_text()) if validation_path.exists() else {}
        )

    def predict(
        self,
        source: ImageSource,
        *,
        confidence: Optional[float] = None,
        iou: float = 0.45,
    ) -> Result:
        image = source.copy() if isinstance(source, Image.Image) else Image.open(source)
        image = image.convert("RGB")
        pixels, metadata = self.preprocess(image, self.model.config.image_size)
        selected_confidence = float(
            confidence
            if confidence is not None
            else self.validation.get("best_confidence", 0.25)
        )
        with self.torch.inference_mode():
            prediction = self.model.predict(
                pixels.unsqueeze(0).to(self.device),
                objectness_threshold=selected_confidence,
                iou_threshold=iou,
                postprocess_on_cpu=self.device.type == "mps",
            )[0]
        boxes = self.restore_boxes(prediction["boxes"].cpu(), metadata)
        return Result(
            image=image,
            boxes=[tuple(float(value) for value in box) for box in boxes],
            scores=[float(value) for value in prediction["scores"].cpu()],
            labels=[int(value) for value in prediction["labels"].cpu()],
            names=self.names,
        )

    def train(self, **options) -> Path:
        from ..training import FineTuner

        return FineTuner(self).run(**options)

    def serve(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: Optional[str] = None,
        jobs_root: Union[str, Path] = "runs/service",
    ) -> None:
        from ..serving import serve_local

        serve_local(
            self,
            host=host,
            port=port,
            api_key=api_key,
            jobs_root=jobs_root,
        )
