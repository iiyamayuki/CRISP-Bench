"""Render marked benchmark images directly from a marks.jsonl manifest."""

from __future__ import annotations

import argparse
import os
from typing import Any

import cv2
from tqdm import tqdm

from data_preprocessing.marks_core import build_output_image_path, ensure_dir, render_mark
from utils.io import read_jsonl
from utils.paths import resolve_path


def _resolve_output_image_path(entry: dict[str, Any], output_dir: str) -> str:
    dataset = entry.get("dataset")
    sample_data_token = entry.get("sample_data_token")
    if not dataset or not sample_data_token:
        raise ValueError("Manifest entry must contain dataset and sample_data_token")
    return os.path.abspath(
        build_output_image_path(dataset, output_dir, sample_data_token, scene_id=entry.get("scene_id"))
    )


def render_manifest_entry(
    entry: dict[str, Any],
    dataroot: str,
    output_dir: str,
    *,
    skip_existing: bool = False,
    strict_image_size: bool = False,
) -> str:
    """Render one manifest row and return the absolute output image path."""
    source_image = entry.get("source_image")
    if not source_image:
        raise ValueError("Manifest entry must contain source_image")

    abs_source_image = resolve_path(source_image, dataroot)
    abs_output_image = _resolve_output_image_path(entry, output_dir)

    if skip_existing and os.path.exists(abs_output_image):
        return abs_output_image

    img = cv2.imread(abs_source_image)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {abs_source_image}")

    image_height, image_width = img.shape[:2]
    expected_size = entry.get("image_size", {})
    expected_width = expected_size.get("width")
    expected_height = expected_size.get("height")
    if expected_width is not None and expected_height is not None:
        if image_width != int(expected_width) or image_height != int(expected_height):
            message = (
                f"Image size mismatch for {source_image}: "
                f"expected {expected_width}x{expected_height}, got {image_width}x{image_height}"
            )
            if strict_image_size:
                raise ValueError(message)
            print(f"[Warning] {message}")

    for mark in entry.get("marks", []):
        img = render_mark(img, mark)

    ensure_dir(os.path.dirname(abs_output_image))
    if not cv2.imwrite(abs_output_image, img):
        raise OSError(f"Failed to write image: {abs_output_image}")
    return abs_output_image


def render_marks_file(
    marks_path: str,
    dataroot: str,
    output_dir: str,
    *,
    skip_existing: bool = False,
    strict_image_size: bool = False,
) -> dict[str, int]:
    """Render every manifest row from *marks_path* and return summary stats."""
    entries = read_jsonl(marks_path)
    stats = {"rendered": 0, "failed": 0}
    for entry in tqdm(entries, desc="Rendering marked images"):
        try:
            render_manifest_entry(
                entry,
                dataroot,
                output_dir,
                skip_existing=skip_existing,
                strict_image_size=strict_image_size,
            )
            stats["rendered"] += 1
        except Exception as exc:
            print(f"[Error] {exc}")
            stats["failed"] += 1
    return stats


def main() -> None:
    """CLI entry point for rendering marked images from marks.jsonl."""
    parser = argparse.ArgumentParser(description="Render marked images from a marks.jsonl manifest.")
    parser.add_argument("--marks", required=True, help="Path to marks.jsonl.")
    parser.add_argument("--dataroot", required=True, help="Dataset root used to resolve source_image paths.")
    parser.add_argument("--output_dir", required=True, help="Directory where marked images will be written.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip output images that already exist.")
    parser.add_argument(
        "--strict_image_size",
        action="store_true",
        help="Fail when a source image size does not match the size recorded in the manifest.",
    )
    args = parser.parse_args()

    stats = render_marks_file(
        args.marks,
        args.dataroot,
        args.output_dir,
        skip_existing=args.skip_existing,
        strict_image_size=args.strict_image_size,
    )
    print(f"[Result] Rendering complete. Stats: {stats}")


if __name__ == "__main__":
    main()