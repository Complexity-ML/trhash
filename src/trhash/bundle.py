"""Resolve local or Hugging Face portable model bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def resolve_bundle(
    model: Union[str, Path],
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    path = Path(model).expanduser()
    if path.is_dir():
        bundle = path.resolve()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError('Hub bundles require `pip install "trhash[runtime]"`') from error
        bundle = Path(
            snapshot_download(
                repo_id=str(model),
                revision=revision,
                token=token,
                allow_patterns=("model.onnx", "trhash.json"),
            )
        )
    for filename in ("model.onnx", "trhash.json"):
        if not (bundle / filename).is_file():
            raise FileNotFoundError(f"portable model bundle is missing {filename}: {bundle}")
    return bundle
