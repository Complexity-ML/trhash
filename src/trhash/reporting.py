"""Render and report image or video predictions for the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from .classification import ClassificationResult
from .result import Result
from .video import VIDEO_EXTENSIONS, VideoWriter, is_video_source


def _is_enabled(value: str) -> bool:
    return value.casefold() in {"1", "true", "yes"}


def _video_output(source, save: str, task: str) -> Path:
    if _is_enabled(save):
        value = str(source)
        stem = Path(value).stem if not value.isdecimal() else "stream"
        return Path("runs") / task / f"{stem or 'stream'}.mp4"
    output = Path(save).expanduser()
    if output.suffix.casefold() in VIDEO_EXTENSIONS:
        return output
    value = str(source)
    stem = Path(value).stem if not value.isdecimal() else "stream"
    return output / f"{stem or 'stream'}.mp4"


def _emit_video(
    results: Iterable,
    *,
    source,
    save: Optional[str],
    stream: bool,
    task: str,
) -> None:
    payloads = []
    writer = None
    output = _video_output(source, save, task) if save is not None else None
    try:
        for result in results:
            payload = result.to_dict()
            if output is not None:
                if writer is None:
                    writer = VideoWriter(
                        output,
                        fps=result.fps or 30.0,
                        size=result.image.size,
                    )
                writer.write(result.plot())
                payload["saved"] = str(output.expanduser().resolve())
            if stream:
                print(json.dumps(payload))
            else:
                payloads.append(payload)
    finally:
        if writer is not None:
            writer.close()
        close = getattr(results, "close", None)
        if close is not None:
            close()
    if not stream:
        print(json.dumps(payloads, indent=2))


def emit_predictions(
    result,
    *,
    source,
    save: Optional[str],
    stream: bool,
    task: str = "predict",
) -> None:
    single_result = isinstance(result, (Result, ClassificationResult))
    if is_video_source(source):
        results = (result,) if single_result else result
        _emit_video(results, source=source, save=save, stream=stream, task=task)
        return
    if single_result:
        payload = result.to_dict()
        if save is not None:
            output = (
                Path("runs") / task / f"{Path(str(source)).stem}.jpg"
                if _is_enabled(save)
                else Path(save)
            )
            payload["saved"] = str(result.save(output))
        print(json.dumps(payload, indent=2))
        return

    output_directory = None
    if save is not None:
        output_directory = Path("runs") / task if _is_enabled(save) else Path(save)
    payloads = []
    for index, item in enumerate(result):
        payload = item.to_dict()
        if output_directory is not None:
            name = Path(item.source).stem if item.source else f"prediction_{index:06d}"
            payload["saved"] = str(item.save(output_directory / f"{name}.jpg"))
        if stream:
            print(json.dumps(payload))
        else:
            payloads.append(payload)
    if not stream:
        print(json.dumps(payloads, indent=2))
