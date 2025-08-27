"""Command line interface for COCO dataset merging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .merger import merge_datasets
try:
    # Prefer yolococo version if installed
    from yolococo import __version__ as YOLOCOCO_VERSION
except Exception:  # pragma: no cover - best-effort version display
    YOLOCOCO_VERSION = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple COCO datasets into one"
    )
    if YOLOCOCO_VERSION:
        parser.add_argument("--version", action="version", version=f"coco-merge (yolococo {YOLOCOCO_VERSION})")
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="Paths to COCO instances JSON files"
    )
    parser.add_argument("--out", required=True, help="Output merged JSON path")
    parser.add_argument(
        "--prefix-mode",
        choices=["none", "basename", "custom"],
        default="none",
        help="How to prefix image file_name to avoid name clashes",
    )
    parser.add_argument(
        "--custom-prefixes",
        nargs="*",
        default=None,
        help="Custom prefixes (one per input) if --prefix-mode=custom",
    )
    parser.add_argument(
        "--align-by-name",
        action="store_true",
        help="If category IDs differ but names match, align to the first dataset's IDs by name",
    )
    parser.add_argument(
        "--drop-duplicate-filenames",
        action="store_true",
        help=(
            "If multiple images share the same file_name post-prefixing, keep the first and drop others"
        ),
    )
    return parser


def main(args: Sequence[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(args=args)
    input_paths = [Path(p) for p in ns.inputs]
    merged = merge_datasets(
        input_paths,
        prefix_mode=ns.prefix_mode,
        custom_prefixes=ns.custom_prefixes,
        align_by_name=ns.align_by_name,
        drop_duplicate_filenames=ns.drop_duplicate_filenames,
    )
    out_path = Path(ns.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(input_paths)} datasets")
    print(
        f"Images: {len(merged['images'])} | Annotations: {len(merged['annotations'])} | Categories: {len(merged['categories'])}"
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
