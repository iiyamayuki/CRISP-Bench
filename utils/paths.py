"""Shared path helpers for dataset assets and task image resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

_DATASET_TO_PROCESSED_DIR = {
    "nuscenes": "nuscenes",
    "scannetpp": "scannetpp",
    "scannet++": "scannetpp",
}


def _normalize_dataset_name(dataset: str | None) -> str | None:
    if not dataset:
        return None
    normalized = str(dataset).strip().lower().replace(" ", "")
    return _DATASET_TO_PROCESSED_DIR.get(normalized)


def make_relative(abs_path: str, dataroot: str) -> str:
    """Return *abs_path* as a path relative to *dataroot*."""
    root = os.path.abspath(dataroot)
    target = os.path.abspath(abs_path)
    if os.path.commonpath([root, target]) != root:
        raise ValueError(f"Path is not under dataroot: {target} (dataroot: {root})")
    return os.path.relpath(target, root)


def resolve_path(rel_path: str, dataroot: str) -> str:
    """Join a dataroot-relative path with *dataroot* to get an absolute path."""
    return os.path.join(dataroot, rel_path)


def resolve_image_path(
    doc: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> str:
    """Resolve a task image path under the current processed dataset layout."""
    image_path = doc.get("image")
    if not image_path:
        raise FileNotFoundError("Image path is missing from the task document")

    image_path_obj = Path(image_path)
    if image_path_obj.is_absolute():
        if image_path_obj.exists():
            return str(image_path_obj)
        raise FileNotFoundError(f"Image not found at: {image_path_obj}")

    repo_root_path = Path(repo_root) if repo_root is not None else REPO_ROOT
    repo_root_path = repo_root_path.resolve()

    if image_path_obj.parts[:2] == ("data", "processed"):
        resolved = repo_root_path / image_path_obj
    else:
        dataset = _normalize_dataset_name(doc.get("meta", {}).get("dataset"))
        if dataset is None:
            raise FileNotFoundError(
                f"Unable to resolve image path for unsupported dataset: {doc.get('meta', {}).get('dataset')!r}"
            )
        resolved = repo_root_path / "data" / "processed" / dataset / image_path_obj

    if resolved.exists():
        return str(resolved)

    raise FileNotFoundError(f"Image not found at: {resolved}")
