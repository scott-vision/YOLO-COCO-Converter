"""Unified CLI for YOLO↔COCO conversion and dataset merging.

Subcommands:
- yolo2coco: Convert YOLO txt labels to COCO JSON
- coco2yolo: Convert COCO JSON to YOLO txt labels
- merge: Merge multiple COCO datasets (delegates to coco_merge)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .convert import yolo_to_coco, coco_to_yolo_files
from . import __version__
from coco_merge.merger import merge_datasets


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="YOLO↔COCO conversion and tools")
    ap.add_argument("--version", action="version", version=f"yolococo {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # yolo2coco
    s1 = sub.add_parser("yolo2coco", help="Convert YOLO txt labels to COCO JSON")
    s1.add_argument("--images", type=Path, required=True, help="Directory with images")
    s1.add_argument("--labels", type=Path, required=True, help="Directory with YOLO .txt files")
    s1.add_argument("--classes", type=Path, default=None, help="Path to classes.txt (optional)")
    s1.add_argument(
        "--sizes",
        type=Path,
        default=None,
        help="CSV: filename,width,height (optional if Pillow installed)",
    )
    s1.add_argument("--out", type=Path, required=True, help="Output COCO JSON path")
    s1.add_argument(
        "--bbox-round",
        type=int,
        default=2,
        help="Round COCO bbox/area to N decimals (set <0 to disable)",
    )
    s1.add_argument(
        "--file-name-mode",
        choices=["name", "relative"],
        default="name",
        help="COCO image file_name mode: basename or path relative to --images",
    )

    # coco2yolo
    s2 = sub.add_parser("coco2yolo", help="Convert COCO JSON to YOLO txt labels")
    s2.add_argument("--coco", type=Path, required=True, help="Path to COCO instances JSON")
    s2.add_argument(
        "--out-labels",
        type=Path,
        required=True,
        help="Output directory for YOLO .txt labels",
    )
    s2.add_argument("--out-classes", type=Path, required=True, help="Output classes.txt")
    s2.add_argument(
        "--keep-category-ids",
        action="store_true",
        help=(
            "Use COCO category ids as YOLO class indices (may be sparse). "
            "Default remaps to 0..K-1."
        ),
    )
    s2.add_argument(
        "--skip-empty-labels",
        action="store_true",
        help="Do not write .txt files for images with no annotations",
    )

    # merge
    s3 = sub.add_parser("merge", help="Merge multiple COCO datasets into one")
    s3.add_argument("--inputs", nargs="+", required=True, help="Paths to COCO instances JSON files")
    s3.add_argument("--out", required=True, help="Output merged JSON path")
    s3.add_argument(
        "--prefix-mode",
        choices=["none", "basename", "custom"],
        default="none",
        help="How to prefix image file_name to avoid name clashes",
    )
    s3.add_argument(
        "--custom-prefixes",
        nargs="*",
        default=None,
        help="Custom prefixes (one per input) if --prefix-mode=custom",
    )
    s3.add_argument(
        "--align-by-name",
        action="store_true",
        help="If category IDs differ but names match, align to the first dataset's IDs by name",
    )
    s3.add_argument(
        "--drop-duplicate-filenames",
        action="store_true",
        help=(
            "If multiple images share the same file_name post-prefixing, keep the first and drop others"
        ),
    )

    return ap


def main(args: Sequence[str] | None = None) -> None:
    ap = build_parser()
    ns = ap.parse_args(args=args)

    try:
        if ns.cmd == "yolo2coco":
            coco = yolo_to_coco(
                ns.images,
                ns.labels,
                ns.classes,
                ns.sizes,
                bbox_round=ns.bbox_round,
                file_name_mode=ns.file_name_mode,
            )
            ns.out.parent.mkdir(parents=True, exist_ok=True)
            with ns.out.open("w", encoding="utf-8") as f:
                json.dump(coco, f, ensure_ascii=False, indent=2)
            print(f"Wrote COCO annotations to: {ns.out}")
            return

        if ns.cmd == "coco2yolo":
            coco_to_yolo_files(
                ns.coco,
                ns.out_labels,
                ns.out_classes,
                keep_category_ids=ns.keep_category_ids,
                skip_empty_labels=ns.skip_empty_labels,
            )
            print(f"Wrote YOLO labels to: {ns.out_labels}")
            print(f"Wrote YOLO classes to: {ns.out_classes}")
            return

        if ns.cmd == "merge":
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
            return
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
