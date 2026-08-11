"""FastAPI application for portable TR-Hash model bundles."""

import io
import time
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from ..backends.onnx import OnnxBackend


def create_app(
    model: Union[str, Path],
    *,
    device: Optional[str] = None,
    api_key: Optional[str] = None,
):
    try:
        from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
        from fastapi.concurrency import run_in_threadpool
    except ImportError as error:
        raise RuntimeError('server dependencies require `pip install "trhash[serve]"`') from error
    backend = OnnxBackend(model, device=device)
    app = FastAPI(title="TR-Hash Vision Server", version="0.1.0")

    def authenticate(x_api_key: Optional[str] = Header(default=None)) -> None:
        if api_key and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": backend.model_id, "providers": backend.providers}

    @app.get("/v1/model", dependencies=[Depends(authenticate)])
    def model_info():
        return {
            "model": backend.model_id,
            "providers": backend.providers,
            "metadata": backend.metadata.__dict__,
        }

    @app.post("/v1/predict", dependencies=[Depends(authenticate)])
    async def predict(
        file: UploadFile = File(...),
        confidence: Optional[float] = None,
        iou_threshold: float = 0.45,
    ):
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception as error:
            raise HTTPException(status_code=400, detail="invalid image") from error
        started = time.perf_counter()
        result = await run_in_threadpool(
            backend.predict,
            image,
            confidence=confidence,
            iou=iou_threshold,
        )
        return {
            **result.to_dict(),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    return app
