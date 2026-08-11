"""Public TR-Hash Vision SDK."""

from typing import TYPE_CHECKING

from .model import Vision
from .benchmarking import BenchmarkEntry, BenchmarkReport
from .result import Result
from .validation import ValidationMetrics

if TYPE_CHECKING:
    from .tracking import ByteTracker

__all__ = [
    "BenchmarkEntry",
    "BenchmarkReport",
    "ByteTracker",
    "Result",
    "ValidationMetrics",
    "Vision",
]
__version__ = "0.2.0.dev0"


def __getattr__(name: str):
    if name == "ByteTracker":
        try:
            from .tracking import ByteTracker
        except ImportError as error:
            raise RuntimeError(
                'tracking requires `pip install "trhash[tracking]"`'
            ) from error
        return ByteTracker
    raise AttributeError(name)
