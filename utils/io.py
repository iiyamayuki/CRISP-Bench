"""Shared JSON and JSONL IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _coerce_path(path: str | Path) -> Path:
    return Path(path)


def _display_path(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return the top-level dict."""
    target = _coerce_path(path)
    try:
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {_display_path(target)}") from exc

    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level JSON object in {_display_path(target)}")
    return data


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write *data* as JSON to *path*, creating parent directories as needed."""
    target = _coerce_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts (one per non-empty line)."""
    target = _coerce_path(path)
    try:
        with target.open("r", encoding="utf-8") as handle:
            rows = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Expected JSON object at line {line_number} in {_display_path(target)}"
                    )
                rows.append(item)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSONL file not found: {_display_path(target)}") from exc

    return rows


def write_jsonl(data: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts as JSONL to *path*, one JSON object per line."""
    target = _coerce_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise TypeError(
                    f"write_jsonl expected dict items, got {type(item).__name__} at index {index}"
                )
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")
