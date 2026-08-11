"""Public `trhash` command-line interface."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from .arguments import assignments
from .commands import HANDLERS

USAGE = """TR-Hash Vision

Usage:
  trhash predict model=MODEL source=IMAGE [confidence=0.25] [save=OUTPUT]
  trhash train model=MODEL data=DATASET.yaml [epochs=20] [batch=16] [device=cuda]
  trhash export model=MODEL [output=runs/export] [opset=18]
  trhash publish bundle=BUNDLE repo=ORG/MODEL [private=true]
  trhash serve model=MODEL [host=127.0.0.1] [port=8000]
  trhash info model=MODEL

MODEL may be a local checkpoint directory or a Hugging Face model ID.
"""


def main(argv: Optional[Sequence[str]] = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return
    command, raw_options = values[0], values[1:]
    if command not in HANDLERS:
        raise SystemExit(f"unknown command: {command}\n\n{USAGE}")
    try:
        HANDLERS[command](assignments(raw_options))
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
