"""Command handlers kept separate from CLI parsing."""

from __future__ import annotations

import json
import os
from typing import Dict

from .arguments import optional_bool, reject_unknown, require
from .model import Vision
from .publishing import publish_bundle
from .reporting import emit_predictions


def vision(options: Dict[str, str]) -> Vision:
    return Vision(
        require(options, "model"),
        endpoint=options.pop("endpoint", None),
        api_key=options.pop("api_key", os.environ.get("TR_HASH_API_KEY")),
        device=options.pop("device", None),
        revision=options.pop("revision", None),
        token=options.pop("token", None),
        runtime=options.pop("runtime", "auto"),
    )


def predict(options: Dict[str, str]) -> None:
    source = require(options, "source")
    confidence = options.pop("confidence", options.pop("conf", None))
    iou = float(options.pop("iou", "0.45"))
    batch = int(options.pop("batch", "1"))
    stream = optional_bool(options, "stream", False)
    save = options.pop("save", None)
    model = vision(options)
    reject_unknown(options)
    result = model.predict(
        source,
        confidence=float(confidence) if confidence is not None else None,
        iou=iou,
        batch=batch,
        stream=stream,
    )
    emit_predictions(result, source=source, save=save, stream=stream)


def track(options: Dict[str, str]) -> None:
    source = require(options, "source")
    tracking = {
        "high_threshold": float(options.pop("high_threshold", "0.5")),
        "low_threshold": float(options.pop("low_threshold", "0.1")),
        "match_iou_threshold": float(options.pop("match_iou_threshold", "0.3")),
        "second_match_iou_threshold": float(
            options.pop("second_match_iou_threshold", "0.2")
        ),
        "track_buffer": int(options.pop("track_buffer", "30")),
        "iou": float(options.pop("iou", "0.45")),
        "batch": int(options.pop("batch", "1")),
        "stream": optional_bool(options, "stream", True),
    }
    new_track_threshold = options.pop("new_track_threshold", None)
    tracking["new_track_threshold"] = (
        float(new_track_threshold) if new_track_threshold is not None else None
    )
    save = options.pop("save", None)
    model = vision(options)
    reject_unknown(options)
    result = model.track(source, **tracking)
    emit_predictions(
        result,
        source=source,
        save=save,
        stream=tracking["stream"],
        task="track",
    )


def val(options: Dict[str, str]) -> None:
    validation = {
        "data": require(options, "data"),
        "confidence": float(options.pop("confidence", options.pop("conf", "0.001"))),
        "iou": float(options.pop("iou", "0.45")),
        "match_iou": float(options.pop("match_iou", "0.50")),
        "batch": int(options.pop("batch", "16")),
    }
    max_images = options.pop("max_images", None)
    validation["max_images"] = int(max_images) if max_images is not None else None
    model = vision(options)
    reject_unknown(options)
    print(json.dumps(model.val(**validation).to_dict(), indent=2))


def train(options: Dict[str, str]) -> None:
    training = {
        "data": require(options, "data"),
        "output": options.pop("output", "runs/train"),
        "epochs": int(options.pop("epochs", "20")),
        "batch": int(options.pop("batch", "16")),
        "workers": int(options.pop("workers", "0")),
        "lr": float(options.pop("lr", "0.01")),
        "expert_lr_multiplier": float(options.pop("expert_lr_multiplier", "1.5")),
        "augmentation": options.pop("augmentation", "strong"),
        "seed": int(options.pop("seed", "42")),
        "resume": optional_bool(options, "resume", False),
    }
    training_device = options.get("device")
    model = vision(options)
    training["device"] = training_device
    reject_unknown(options)
    print(json.dumps({"checkpoint": str(model.train(**training))}, indent=2))


def export(options: Dict[str, str]) -> None:
    export_format = options.pop("format", "onnx")
    exporting = {
        "output": options.pop("output", "runs/export"),
        "format": export_format,
        "opset": int(options.pop("opset", "18")),
        "verify": optional_bool(options, "verify", True),
    }
    if export_format in {"coreml", "tensorrt"}:
        exporting["precision"] = options.pop("precision", "fp16")
    if export_format == "tensorrt":
        exporting["max_batch"] = int(options.pop("max_batch", "32"))
        exporting["workspace_gb"] = float(options.pop("workspace_gb", "1"))
    model = vision(options)
    reject_unknown(options)
    print(json.dumps({"bundle": str(model.export(**exporting))}, indent=2))


def benchmark(options: Dict[str, str]) -> None:
    source = require(options, "source")
    benchmarking = {
        "formats": tuple(
            value.strip() for value in options.pop("formats", "onnx,torchscript").split(",")
        ),
        "output": options.pop("output", "runs/benchmark"),
        "warmup": int(options.pop("warmup", "3")),
        "runs": int(options.pop("runs", "20")),
        "batch": int(options.pop("batch", "1")),
        "device": options.pop("benchmark_device", None),
    }
    model = vision(options)
    reject_unknown(options)
    print(json.dumps(model.benchmark(source, **benchmarking).to_dict(), indent=2))


def _comma_values(options: Dict[str, str], name: str, cast):
    value = options.pop(name, None)
    return None if value is None else tuple(cast(item.strip()) for item in value.split(","))


def tune(options: Dict[str, str]) -> None:
    space = {}
    for public_name, parameter_name, cast in (
        ("lrs", "lr", float),
        ("expert_lr_multipliers", "expert_lr_multiplier", float),
        ("augmentations", "augmentation", str),
    ):
        values = _comma_values(options, public_name, cast)
        if values is not None:
            space[parameter_name] = values
    tuning = {
        "data": require(options, "data"),
        "output": options.pop("output", "runs/tune"),
        "iterations": int(options.pop("iterations", "10")),
        "epochs": int(options.pop("epochs", "10")),
        "batch": int(options.pop("batch", "16")),
        "workers": int(options.pop("workers", "0")),
        "seed": int(options.pop("seed", "42")),
        "resume": optional_bool(options, "resume", False),
        "space": space or None,
    }
    max_images = options.pop("max_images", None)
    tuning["max_images"] = int(max_images) if max_images is not None else None
    tuning_device = options.get("device")
    model = vision(options)
    tuning["device"] = tuning_device
    reject_unknown(options)
    print(json.dumps(model.tune(**tuning).to_dict(), indent=2))


def serve(options: Dict[str, str]) -> None:
    serving = {
        "host": options.pop("host", "127.0.0.1"),
        "port": int(options.pop("port", "8000")),
        "jobs_root": options.pop("jobs_root", "runs/service"),
    }
    api_key = options.get("api_key", os.environ.get("TR_HASH_API_KEY"))
    model = vision(options)
    serving["api_key"] = api_key
    reject_unknown(options)
    model.serve(**serving)


def publish(options: Dict[str, str]) -> None:
    bundle = require(options, "bundle")
    repo = require(options, "repo")
    private = optional_bool(options, "private", True)
    token = options.pop("token", None)
    reject_unknown(options)
    oid = publish_bundle(bundle, repo, private=private, token=token)
    print(json.dumps({"repo": repo, "commit": oid}, indent=2))


def info(options: Dict[str, str]) -> None:
    model = vision(options)
    reject_unknown(options)
    backend = model.backend
    metadata = getattr(backend, "metadata", None)
    payload = {
        "model": backend.model_id,
        "backend": type(backend).__name__,
        "endpoint": getattr(backend, "endpoint", None),
        "bundle": str(getattr(backend, "bundle", "")) or None,
        "checkpoint": str(getattr(backend, "checkpoint", "")) or None,
        "providers": getattr(backend, "providers", None),
        "task": getattr(metadata, "task", getattr(backend, "task", None)),
    }
    print(json.dumps(payload, indent=2))


HANDLERS = {
    "predict": predict,
    "track": track,
    "val": val,
    "train": train,
    "sft": train,
    "export": export,
    "benchmark": benchmark,
    "tune": tune,
    "publish": publish,
    "serve": serve,
    "info": info,
}
