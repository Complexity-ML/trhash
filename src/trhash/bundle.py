"""Resolve local or Hugging Face portable model bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .metadata import ModelMetadata


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
                allow_patterns=("trhash.json", "*.onnx", "*.torchscript"),
            )
        )
    if not (bundle / "trhash.json").is_file():
        raise FileNotFoundError(f"portable model bundle is missing trhash.json: {bundle}")
    metadata = ModelMetadata.load(bundle)
    if not (bundle / metadata.model_file).is_file():
        raise FileNotFoundError(
            f"portable model bundle is missing {metadata.model_file}: {bundle}"
        )
    return bundle
