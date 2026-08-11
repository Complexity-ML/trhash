"""Small `key=value` parser shared by CLI commands."""

from __future__ import annotations

from typing import Dict, Iterable


def assignments(values: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected key=value, received: {value}")
        key, item = value.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key or not item:
            raise ValueError(f"invalid key=value argument: {value}")
        if key in parsed:
            raise ValueError(f"duplicate argument: {key}")
        parsed[key] = item
    return parsed


def require(options: Dict[str, str], name: str) -> str:
    try:
        return options.pop(name)
    except KeyError as error:
        raise ValueError(f"missing required argument: {name}=...") from error


def optional_bool(options: Dict[str, str], name: str, default: bool = False) -> bool:
    raw = options.pop(name, str(default)).casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def reject_unknown(options: Dict[str, str]) -> None:
    if options:
        raise ValueError(f"unknown arguments: {', '.join(sorted(options))}")
