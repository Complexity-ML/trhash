"""Optional PyTorch backend loaded only for local execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from ..metadata import metadata_from_checkpoint
from ..runtime import imports, resolve_checkpoint, resolve_device
from .portable import PortableDetectionBackend


class LocalBackend(PortableDetectionBackend):
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
            load_vision_task_checkpoint,
        ) = imports()
        self.checkpoint = resolve_checkpoint(
            model,
            revision=revision,
            token=token,
        )
        self.model_id = str(model)
        self.device = resolve_device(self.torch, device)
        task_path = self.checkpoint / "vision_task.json"
        if task_path.is_file():
            manifest = json.loads(task_path.read_text())
            self.task = str(manifest.get("task"))
            self.model = load_vision_task_checkpoint(self.checkpoint, device=self.device)
        else:
            self.task = "detection"
            self.model = load_checkpoint(self.checkpoint, device=self.device)
        names_path = self.checkpoint / "class_names.json"
        if not names_path.is_file():
            # Interrupted fine-tunes leave shared metadata at the run root.
            names_path = self.checkpoint.parent / "class_names.json"
        self.names = (
            tuple(str(name) for name in json.loads(names_path.read_text()))
            if names_path.exists()
            else tuple(str(index) for index in range(self._num_classes()))
        )
        if len(self.names) != self._num_classes():
            raise ValueError("class_names.json does not match the model class count")
        validation_path = self.checkpoint / "validation.json"
        self.validation = (
            json.loads(validation_path.read_text()) if validation_path.exists() else {}
        )
        self.metadata = metadata_from_checkpoint(self)

    def _num_classes(self) -> int:
        if self.task == "depth":
            return 0
        if self.task == "classification":
            return int(self.model.head.out_features)
        if self.task == "semantic_segmentation":
            return int(self.model.num_classes)
        if self.task == "pose":
            return int(self.model.num_keypoints)
        return int(self.model.config.num_classes)

    def _predict_raw(self, pixels):
        with self.torch.inference_mode():
            values = self.torch.from_numpy(pixels).to(self.device)
            if self.task == "classification":
                return self.model(values)["logits"].cpu().numpy()
            if self.task == "semantic_segmentation":
                return self.model(values)["logits"].cpu().numpy()
            if self.task == "depth":
                return self.model(values)["depth"].cpu().numpy()
            if self.task == "pose":
                return self.model(values)["heatmaps"].cpu().numpy()
            if self.task == "instance_segmentation":
                outputs = self.model.forward_instance(values)
                return tuple(
                    outputs[name].cpu().numpy()
                    for name in ("raw", "mask_coefficients", "prototypes")
                )
            if self.task == "obb":
                outputs = self.model.forward_obb(values)
                return outputs["raw"].cpu().numpy(), outputs["angles"].cpu().numpy()
            if bool(getattr(self.model.config, "end_to_end", False)):
                _, one_to_one = self.model.forward_branches(values)
                if one_to_one is None:
                    raise RuntimeError("NMS-free detector did not return its one-to-one branch")
                return one_to_one.cpu().numpy()
            return self.model.forward_predictions(values).cpu().numpy()

    def train(self, **options) -> Path:
        if self.task != "detection":
            raise NotImplementedError(
                f"fine-tuning is not implemented for task={self.task}"
            )
        from ..training import FineTuner

        return FineTuner(self).run(**options)

    def export(self, **options) -> Path:
        from ..exporter import export_model

        return export_model(self, **options)

    def serve(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: Optional[str] = None,
        jobs_root: Union[str, Path] = "runs/service",
    ) -> None:
        from ..server.runner import run_server

        bundle = self.export(output=Path(jobs_root) / "bundle")
        runtime_device = {
            "cuda": "cuda",
            "mps": "coreml",
        }.get(self.device.type, "cpu")
        run_server(
            bundle,
            device=runtime_device,
            host=host,
            port=port,
            api_key=api_key,
        )
