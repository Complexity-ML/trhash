"""Comparable end-to-end benchmarks for portable model formats."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from .sources import ImageSource, expand_sources


@dataclass(frozen=True)
class BenchmarkEntry:
    format: str
    backend: str
    providers: tuple[str, ...]
    batch: int
    runs: int
    latency_ms: float
    p50_ms: float
    p95_ms: float
    throughput_images_s: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    bundle_size_mb: float


@dataclass(frozen=True)
class BenchmarkReport:
    entries: tuple[BenchmarkEntry, ...]

    @property
    def fastest(self) -> str:
        return min(self.entries, key=lambda entry: entry.latency_ms).format

    def to_dict(self):
        return {
            "fastest": self.fastest,
            "results": [asdict(entry) for entry in self.entries],
        }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bundle_size_mb(bundle: Path) -> float:
    return sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file()) / 1e6


def benchmark_model(
    model,
    source: ImageSource,
    *,
    formats: Sequence[str] = ("onnx", "torchscript"),
    output: Union[str, Path] = "runs/benchmark",
    warmup: int = 3,
    runs: int = 20,
    batch: int = 1,
    device: Optional[str] = None,
) -> BenchmarkReport:
    if warmup < 0 or runs <= 0 or batch <= 0:
        raise ValueError("warmup must be non-negative; runs and batch must be positive")
    sources, single = expand_sources(source)
    if not single:
        raise ValueError("benchmark source must be one image, not a directory or iterable")
    item = next(sources)
    if isinstance(formats, str):
        formats = tuple(value.strip() for value in formats.split(","))
    selected_formats = tuple(dict.fromkeys(value.casefold() for value in formats if value))
    if not selected_formats:
        raise ValueError("at least one benchmark format is required")

    from .model import Vision

    output_path = Path(output).expanduser().resolve()
    entries = []
    for export_format in selected_formats:
        bundle = model.export(format=export_format, output=output_path / export_format)
        runtime_model = Vision(bundle, device=device)
        try:
            inputs = [item] * batch
            for _ in range(warmup):
                runtime_model.predict(inputs, batch=batch)

            latencies = []
            stage_speeds = {"preprocess": [], "inference": [], "postprocess": []}
            for _ in range(runs):
                started = time.perf_counter()
                results = runtime_model.predict(inputs, batch=batch)
                latencies.append((time.perf_counter() - started) * 1000.0 / batch)
                speed = results[0].speed
                for stage in stage_speeds:
                    stage_speeds[stage].append(float(speed.get(stage, 0.0)))

            mean_latency = statistics.fmean(latencies)
            providers = tuple(
                str(value) for value in getattr(runtime_model.backend, "providers", ())
            )
            entries.append(
                BenchmarkEntry(
                    format=export_format,
                    backend=type(runtime_model.backend).__name__,
                    providers=providers,
                    batch=batch,
                    runs=runs,
                    latency_ms=mean_latency,
                    p50_ms=_percentile(latencies, 0.50),
                    p95_ms=_percentile(latencies, 0.95),
                    throughput_images_s=1000.0 / mean_latency,
                    preprocess_ms=statistics.fmean(stage_speeds["preprocess"]),
                    inference_ms=statistics.fmean(stage_speeds["inference"]),
                    postprocess_ms=statistics.fmean(stage_speeds["postprocess"]),
                    bundle_size_mb=_bundle_size_mb(bundle),
                )
            )
        finally:
            runtime_model.close()
    return BenchmarkReport(tuple(entries))
