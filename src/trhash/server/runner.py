"""Uvicorn runner kept separate from application construction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def run_server(
    model: Union[str, Path],
    *,
    device: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: Optional[str] = None,
    **_unused,
) -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError('server dependencies require `pip install "trhash[serve]"`') from error
    from .app import create_app

    uvicorn.run(
        create_app(model, device=device, api_key=api_key),
        host=host,
        port=port,
    )


def main() -> None:
    model = os.environ.get("TRHASH_MODEL")
    if not model:
        raise SystemExit("TRHASH_MODEL is required")
    run_server(
        model,
        device=os.environ.get("TRHASH_DEVICE"),
        host=os.environ.get("TRHASH_HOST", "0.0.0.0"),
        port=int(os.environ.get("TRHASH_PORT", "8000")),
        api_key=os.environ.get("TR_HASH_API_KEY"),
    )
