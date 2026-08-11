"""Command handlers kept separate from CLI parsing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from .arguments import optional_bool, reject_unknown, require
from .model import Vision
from .publishing import publish_bundle


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
    save = options.pop("save", None)
    model = vision(options)
    reject_unknown(options)
    result = model.predict(
        source,
        confidence=float(confidence) if confidence is not None else None,
        iou=iou,
    )
    payload = result.to_dict()
    if save is not None:
        output = (
            Path("runs/predict") / f"{Path(source).stem}.jpg"
            if save.casefold() in {"1", "true", "yes"}
            else Path(save)
        )
        payload["saved"] = str(result.save(output))
    print(json.dumps(payload, indent=2))


def train(options: Dict[str, str]) -> None:
    training = {
        "data": require(options, "data"),
        "output": options.pop("output", "runs/train"),
        "epochs": int(options.pop("epochs", "20")),
        "batch": int(options.pop("batch", "16")),
        "workers": int(options.pop("workers", "0")),
        "lr": float(options.pop("lr", "0.01")),
        "expert_lr_multiplier": float(options.pop("expert_lr_multiplier", "1.5")),
        "seed": int(options.pop("seed", "42")),
    }
    training_device = options.get("device")
    model = vision(options)
    training["device"] = training_device
    reject_unknown(options)
    print(json.dumps({"checkpoint": str(model.train(**training))}, indent=2))


def export(options: Dict[str, str]) -> None:
    exporting = {
        "output": options.pop("output", "runs/export"),
        "opset": int(options.pop("opset", "18")),
    }
    model = vision(options)
    reject_unknown(options)
    print(json.dumps({"bundle": str(model.export(**exporting))}, indent=2))


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
    payload = {
        "model": backend.model_id,
        "backend": type(backend).__name__,
        "endpoint": getattr(backend, "endpoint", None),
        "bundle": str(getattr(backend, "bundle", "")) or None,
        "checkpoint": str(getattr(backend, "checkpoint", "")) or None,
        "providers": getattr(backend, "providers", None),
    }
    print(json.dumps(payload, indent=2))


HANDLERS = {
    "predict": predict,
    "train": train,
    "sft": train,
    "export": export,
    "publish": publish,
    "serve": serve,
    "info": info,
}
