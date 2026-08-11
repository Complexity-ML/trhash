"""Public `trhash` command-line interface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from .arguments import assignments, reject_unknown, require
from .model import Vision

USAGE = """TR-Hash Vision

Usage:
  trhash predict model=MODEL source=IMAGE [confidence=0.25] [save=OUTPUT]
  trhash train model=MODEL data=DATASET.yaml [epochs=20] [batch=16] [device=cuda]
  trhash serve model=MODEL [host=127.0.0.1] [port=8000]
  trhash info model=MODEL

MODEL may be a local checkpoint directory or a Hugging Face model ID.
"""


def _vision(options: Dict[str, str]) -> Vision:
    return Vision(
        require(options, "model"),
        endpoint=options.pop("endpoint", None),
        api_key=options.pop("api_key", os.environ.get("TR_HASH_API_KEY")),
        device=options.pop("device", None),
        revision=options.pop("revision", None),
        token=options.pop("token", None),
    )


def _predict(options: Dict[str, str]) -> None:
    source = require(options, "source")
    confidence = options.pop("confidence", options.pop("conf", None))
    iou = float(options.pop("iou", "0.45"))
    save = options.pop("save", None)
    model = _vision(options)
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


def _train(options: Dict[str, str]) -> None:
    data = require(options, "data")
    training = {
        "data": data,
        "output": options.pop("output", "runs/train"),
        "epochs": int(options.pop("epochs", "20")),
        "batch": int(options.pop("batch", "16")),
        "workers": int(options.pop("workers", "0")),
        "lr": float(options.pop("lr", "0.01")),
        "expert_lr_multiplier": float(options.pop("expert_lr_multiplier", "1.5")),
        "seed": int(options.pop("seed", "42")),
    }
    training_device = options.get("device")
    model = _vision(options)
    training["device"] = training_device
    reject_unknown(options)
    checkpoint = model.train(**training)
    print(json.dumps({"checkpoint": str(checkpoint)}, indent=2))


def _serve(options: Dict[str, str]) -> None:
    serving = {
        "host": options.pop("host", "127.0.0.1"),
        "port": int(options.pop("port", "8000")),
        "jobs_root": options.pop("jobs_root", "runs/service"),
    }
    api_key = options.get("api_key", os.environ.get("TR_HASH_API_KEY"))
    model = _vision(options)
    serving["api_key"] = api_key
    reject_unknown(options)
    model.serve(**serving)


def _info(options: Dict[str, str]) -> None:
    model = _vision(options)
    reject_unknown(options)
    backend = model.backend
    payload = {
        "model": backend.model_id,
        "backend": type(backend).__name__,
        "endpoint": getattr(backend, "endpoint", None),
        "checkpoint": str(getattr(backend, "checkpoint", "")) or None,
        "device": str(getattr(backend, "device", "")) or None,
        "parameters": getattr(getattr(backend, "model", None), "num_parameters", lambda: None)(),
    }
    print(json.dumps(payload, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return
    command, raw_options = values[0], values[1:]
    handlers = {"predict": _predict, "train": _train, "sft": _train, "serve": _serve, "info": _info}
    if command not in handlers:
        raise SystemExit(f"unknown command: {command}\n\n{USAGE}")
    try:
        handlers[command](assignments(raw_options))
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
