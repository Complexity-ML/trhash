"""Publish portable bundles to Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .metadata import ModelMetadata


def publish_bundle(
    bundle: Union[str, Path],
    repo_id: str,
    *,
    private: bool = True,
    token: Optional[str] = None,
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("publishing requires huggingface_hub") from error
    bundle_path = Path(bundle).expanduser().resolve()
    metadata = ModelMetadata.load(bundle_path)
    if not (bundle_path / metadata.model_file).is_file():
        raise FileNotFoundError(bundle_path / metadata.model_file)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(bundle_path),
        commit_message="Publish portable TR-Hash ONNX bundle",
    )
    return result.oid
