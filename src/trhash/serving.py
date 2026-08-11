"""Local HTTP server adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def serve_local(
    backend,
    *,
    host: str,
    port: int,
    api_key: Optional[str],
    jobs_root: Union[str, Path],
) -> None:
    try:
        import uvicorn
        from complexity.generative.detection.service import create_app
    except ImportError as error:
        raise RuntimeError('serving requires `pip install "trhash[serve]"`') from error
    uvicorn.run(
        create_app(
            backend.checkpoint,
            device=str(backend.device),
            jobs_root=Path(jobs_root),
            api_key=api_key,
        ),
        host=host,
        port=port,
    )
