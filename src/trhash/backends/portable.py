"""Shared detection pipeline and runtime selection for portable bundles."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image

from ..bundle import resolve_bundle
from ..classification import ClassificationResult
from ..decoding import decode
from ..depth import DepthResult
from ..instance_segmentation import InstanceSegmentationResult
from ..mask_decoding import decode_instance_masks
from ..metadata import ModelMetadata
from ..obb import OBBResult
from ..pose import PoseResult
from ..preprocessing import preprocess, restore_boxes
from ..result import Result
from ..segmentation import SemanticSegmentationResult

ImageSource = Union[str, Path, Image.Image]


class PortableDetectionBackend:
    def predict(
        self,
        source: ImageSource,
        *,
        confidence: Optional[float] = None,
        iou: float = 0.45,
    ) -> Result:
        return self.predict_batch((source,), confidence=confidence, iou=iou)[0]

    def predict_batch(
        self,
        sources: Sequence[ImageSource],
        *,
        confidence: Optional[float] = None,
        iou: float = 0.45,
    ) -> list[Result]:
        started = time.perf_counter()
        images = [
            (source.copy() if isinstance(source, Image.Image) else Image.open(source)).convert("RGB")
            for source in sources
        ]
        prepared = [preprocess(image, self.metadata) for image in images]
        pixels = np.stack([item[0] for item in prepared])
        preprocessed = time.perf_counter()
        predictions = self._predict_raw(pixels)
        inferred = time.perf_counter()
        threshold = (
            float(confidence)
            if confidence is not None
            else self.metadata.recommended_confidence
        )
        results = []
        if self.metadata.task == "obb":
            if not isinstance(predictions, (tuple, list)) or len(predictions) != 2:
                raise ValueError("OBB runtime must return predictions and angles")
            raw_batch, angles_batch = predictions
            expected_cells = sum(grid * grid for grid in self.metadata.grid_sizes)
            if raw_batch.shape[0] != len(images) or angles_batch.shape != (
                len(images),
                expected_cells,
            ):
                raise ValueError("OBB graph outputs have invalid shapes")
            for source, image, raw, angles, (_, geometry) in zip(
                sources,
                images,
                raw_batch,
                angles_batch,
                prepared,
            ):
                boxes, scores, labels, indices = decode(
                    raw,
                    self.metadata,
                    confidence=threshold,
                    iou=iou,
                    return_indices=True,
                )
                boxes = restore_boxes(boxes, self.metadata, geometry)
                results.append(
                    OBBResult(
                        image=image,
                        boxes=[
                            (*tuple(float(value) for value in box), float(angles[index]))
                            for box, index in zip(boxes, indices)
                        ],
                        scores=[float(value) for value in scores],
                        labels=[int(value) for value in labels],
                        names=self.names,
                        source=None if isinstance(source, Image.Image) else str(source),
                    )
                )
        elif self.metadata.task == "instance_segmentation":
            if not isinstance(predictions, (tuple, list)) or len(predictions) != 3:
                raise ValueError("instance runtime must return three graph outputs")
            raw_batch, coefficients_batch, prototypes_batch = predictions
            expected_cells = sum(grid * grid for grid in self.metadata.grid_sizes)
            num_prototypes = int(self.metadata.task_options["num_prototypes"])
            if raw_batch.shape[0] != len(images):
                raise ValueError("instance predictions have the wrong batch size")
            if coefficients_batch.shape != (
                len(images),
                expected_cells,
                num_prototypes,
            ):
                raise ValueError("instance mask coefficients have an invalid shape")
            if (
                prototypes_batch.ndim != 4
                or prototypes_batch.shape[:2] != (len(images), num_prototypes)
            ):
                raise ValueError("instance prototypes have an invalid shape")
            mask_threshold = float(
                self.metadata.task_options.get("mask_threshold", 0.5)
            )
            for source, image, raw, coefficients, prototypes, (_, geometry) in zip(
                sources,
                images,
                raw_batch,
                coefficients_batch,
                prototypes_batch,
                prepared,
            ):
                boxes, scores, labels, indices = decode(
                    raw,
                    self.metadata,
                    confidence=threshold,
                    iou=iou,
                    return_indices=True,
                )
                masks = decode_instance_masks(
                    coefficients,
                    prototypes,
                    indices,
                    boxes,
                    geometry,
                    image_size=self.metadata.image_size,
                    threshold=mask_threshold,
                )
                boxes = restore_boxes(boxes, self.metadata, geometry)
                results.append(
                    InstanceSegmentationResult(
                        image=image,
                        boxes=[tuple(float(value) for value in box) for box in boxes],
                        masks=masks,
                        scores=[float(value) for value in scores],
                        labels=[int(value) for value in labels],
                        names=self.names,
                        source=None if isinstance(source, Image.Image) else str(source),
                    )
                )
        elif self.metadata.task == "classification":
            if predictions.ndim != 2 or predictions.shape[1] != self.metadata.num_classes:
                raise ValueError("classification output must have shape [batch, classes]")
            shifted = predictions - predictions.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            for source, image, scores in zip(sources, images, probabilities):
                labels = np.argsort(-scores)
                results.append(
                    ClassificationResult(
                        image=image,
                        scores=[float(scores[label]) for label in labels],
                        labels=[int(label) for label in labels],
                        names=self.names,
                        source=None if isinstance(source, Image.Image) else str(source),
                    )
                )
        elif self.metadata.task == "semantic_segmentation":
            expected = (
                len(images),
                self.metadata.num_classes,
                self.metadata.image_size,
                self.metadata.image_size,
            )
            if predictions.shape != expected:
                raise ValueError(
                    "semantic output must have shape [batch, classes, height, width]"
                )
            labels = predictions.argmax(axis=1)
            for source, image, label_map in zip(sources, images, labels):
                mask = Image.fromarray(label_map.astype(np.int32), mode="I").resize(
                    image.size,
                    Image.Resampling.NEAREST,
                )
                results.append(
                    SemanticSegmentationResult(
                        image=image,
                        mask=mask,
                        names=self.names,
                        source=None if isinstance(source, Image.Image) else str(source),
                    )
                )
        elif self.metadata.task == "depth":
            expected = (
                len(images),
                1,
                self.metadata.image_size,
                self.metadata.image_size,
            )
            if predictions.shape != expected:
                raise ValueError("depth output must have shape [batch, 1, height, width]")
            for source, image, depth_map in zip(sources, images, predictions[:, 0]):
                depth = Image.fromarray(depth_map.astype(np.float32), mode="F").resize(
                    image.size,
                    Image.Resampling.BILINEAR,
                )
                results.append(
                    DepthResult(
                        image=image,
                        depth=depth,
                        source=None if isinstance(source, Image.Image) else str(source),
                    )
                )
        elif self.metadata.task == "pose":
            expected = (
                len(images),
                self.metadata.num_classes,
                self.metadata.image_size,
                self.metadata.image_size,
            )
            if predictions.shape != expected:
                raise ValueError(
                    "pose output must have shape [batch, keypoints, height, width]"
                )
            for source, image, heatmaps in zip(sources, images, predictions):
                keypoints = []
                height, width = heatmaps.shape[-2:]
                for heatmap in heatmaps:
                    flat_index = int(heatmap.argmax())
                    row, column = divmod(flat_index, width)
                    keypoints.append(
                        (
                            (column + 0.5) / width * image.width,
                            (row + 0.5) / height * image.height,
                            float(heatmap[row, column]),
                        )
                    )
                results.append(
                    PoseResult(
                        image=image,
                        keypoints=keypoints,
                        names=self.names,
                        source=None if isinstance(source, Image.Image) else str(source),
                    )
                )
        else:
            for source, image, raw, (_, geometry) in zip(
                sources,
                images,
                predictions,
                prepared,
            ):
                boxes, scores, labels = decode(
                    raw,
                    self.metadata,
                    confidence=threshold,
                    iou=iou,
                )
                boxes = restore_boxes(boxes, self.metadata, geometry)
                results.append(
                    Result(
                        image=image,
                        boxes=[tuple(float(value) for value in box) for box in boxes],
                        scores=[float(value) for value in scores],
                        labels=[int(value) for value in labels],
                        names=self.names,
                        source=None if isinstance(source, Image.Image) else str(source),
                    )
                )
        finished = time.perf_counter()
        count = max(len(images), 1)
        speed = {
            "preprocess": (preprocessed - started) * 1000.0 / count,
            "inference": (inferred - preprocessed) * 1000.0 / count,
            "postprocess": (finished - inferred) * 1000.0 / count,
        }
        for result in results:
            result.speed.update(speed)
        return results


def load_portable_backend(
    model,
    *,
    runtime: str = "auto",
    device: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
):
    bundle = resolve_bundle(model, revision=revision, token=token)
    metadata = ModelMetadata.load(bundle)
    if metadata.task not in {
        "detection",
        "classification",
        "semantic_segmentation",
        "depth",
        "pose",
        "instance_segmentation",
        "obb",
    }:
        raise NotImplementedError(
            f"portable runtime is not implemented for task={metadata.task}"
        )
    extension = Path(metadata.model_file).suffix.casefold()
    runtime_by_extension = {
        ".onnx": "onnx",
        ".torchscript": "torchscript",
        ".mlpackage": "coreml",
        ".engine": "tensorrt",
    }
    detected_runtime = runtime_by_extension.get(extension)
    if detected_runtime is None:
        raise ValueError(f"unsupported portable model file: {metadata.model_file}")
    if runtime != "auto" and runtime != detected_runtime:
        raise ValueError(
            f"bundle contains {detected_runtime}, but runtime={runtime} was requested"
        )
    if detected_runtime == "onnx":
        from .onnx import OnnxBackend

        return OnnxBackend(bundle, device=device)
    if detected_runtime == "coreml":
        from .coreml import CoreMLBackend

        return CoreMLBackend(bundle, device=device)
    if detected_runtime == "tensorrt":
        from .tensorrt import TensorRTBackend

        return TensorRTBackend(bundle, device=device)
    from .torchscript import TorchScriptBackend

    return TorchScriptBackend(bundle, device=device)
