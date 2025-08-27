"""YOLO ↔ COCO conversion utilities.

This module provides programmatic functions for converting between YOLO
text-label format and COCO instances JSON format. It intentionally keeps
dependencies minimal; Pillow is optional and used only to infer image sizes
when converting YOLO → COCO if a CSV with sizes is not provided.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def try_pillow_get_size(img_path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image  # optional but installed by default via pyproject
    except ImportError:
        return None
    try:
        with Image.open(img_path) as im:
            return im.width, im.height
    except Exception:
        return None


def load_sizes_csv(csv_path: Path) -> Dict[str, Tuple[int, int]]:
    sizes: Dict[str, Tuple[int, int]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            filename, w, h = row[0], int(row[1]), int(row[2])
            sizes[filename] = (w, h)
    return sizes


def load_classes_txt(classes_path: Optional[Path]) -> List[str]:
    if classes_path and classes_path.exists():
        with classes_path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() != ""]
    return []  # will synthesize names if missing


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def yolo_to_coco(
    images_dir: Path,
    labels_dir: Path,
    classes_path: Optional[Path] = None,
    sizes_csv: Optional[Path] = None,
    *,
    bbox_round: Optional[int] = 2,
    file_name_mode: str = "name",
) -> Dict[str, object]:
    """Convert YOLO txt labels to a COCO dataset dictionary.

    Parameters
    ----------
    images_dir: directory containing image files
    labels_dir: directory containing YOLO .txt files (same stems as images)
    classes_path: optional path to classes.txt (names per line)
    sizes_csv: optional CSV (no header): filename,width,height. Used if Pillow
               is not installed or fails to read a particular image.

    bbox_round:
        Number of decimals to round bbox coordinates and area. Set to a
        negative value (e.g., -1) to disable rounding. Default: 2.
    file_name_mode:
        How to populate COCO images[].file_name: "name" (basename) or
        "relative" (path relative to images_dir, using '/' separators).

    Returns
    -------
    dict: COCO dataset with keys images, annotations, categories
    """

    classes = load_classes_txt(classes_path)
    sizes_map = load_sizes_csv(sizes_csv) if sizes_csv else {}

    # Decide file_name formatter once
    def file_name_for(img_path: Path) -> str:
        if file_name_mode == "relative":
            try:
                rel = img_path.relative_to(images_dir)
            except ValueError:
                # If not under images_dir, fallback to name
                rel = img_path.name
                return str(rel)
            return rel.as_posix()
        return img_path.name

    # Prepare size resolver once (CSV map takes precedence)
    try:
        from PIL import Image  # type: ignore
        pil_available = True
        def _read_img_size(p: Path) -> Optional[Tuple[int, int]]:
            try:
                with Image.open(p) as im:
                    return im.width, im.height
            except Exception:
                return None
    except ImportError:
        pil_available = False
        def _read_img_size(p: Path) -> Optional[Tuple[int, int]]:
            return None

    images: List[dict] = []
    annotations: List[dict] = []
    categories: List[dict] = []
    ann_id = 1
    img_id = 1

    # Build categories (if classes known now)
    if classes:
        for i, name in enumerate(classes):
            categories.append({"id": i, "name": name})

    img_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not img_files:
        raise ValueError(f"No images found under: {images_dir}")

    seen_class_ids = set()

    for img_path in img_files:
        rel_name = file_name_for(img_path)
        # Resolve image size
        if rel_name in sizes_map:
            w, h = sizes_map[rel_name]
        else:
            size = _read_img_size(img_path)
            if size is None:
                raise ValueError(
                    f"Missing image size for {rel_name}. Install Pillow or provide --sizes CSV."
                )
            w, h = size

        images.append({"id": img_id, "file_name": rel_name, "width": w, "height": h})

        # Read matching label file
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # Some YOLO variants include conf as 6th value; ignore if present
                    if len(parts) not in (5, 6):
                        continue
                    try:
                        cls = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        ww = float(parts[3])
                        hh = float(parts[4])
                    except Exception:
                        continue

                    seen_class_ids.add(cls)

                    # convert to COCO xywh (absolute pixels)
                    abs_w = ww * w
                    abs_h = hh * h
                    x_min = (xc * w) - (abs_w / 2.0)
                    y_min = (yc * h) - (abs_h / 2.0)

                    # clip to image bounds
                    x_min = max(0.0, x_min)
                    y_min = max(0.0, y_min)
                    abs_w = max(0.0, min(abs_w, w - x_min))
                    abs_h = max(0.0, min(abs_h, h - y_min))

                    if isinstance(bbox_round, int) and bbox_round >= 0:
                        bbox = [
                            round(x_min, bbox_round),
                            round(y_min, bbox_round),
                            round(abs_w, bbox_round),
                            round(abs_h, bbox_round),
                        ]
                        area = round(abs_w * abs_h, bbox_round)
                    else:
                        bbox = [x_min, y_min, abs_w, abs_h]
                        area = abs_w * abs_h

                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        img_id += 1

    # If classes.txt was missing, synthesize categories from seen ids
    if not categories:
        categories = (
            [{"id": cid, "name": f"class_{cid}"} for cid in sorted(seen_class_ids)]
            if seen_class_ids
            else []
        )

    coco = {"images": images, "annotations": annotations, "categories": categories}
    return coco


def coco_to_yolo_files(
    coco_json: Union[Path, Dict[str, object]],
    out_labels_dir: Path,
    out_classes_path: Path,
    *,
    keep_category_ids: bool = False,
    skip_empty_labels: bool = False,
) -> None:
    """Convert COCO JSON to YOLO .txt files and write classes.txt.

    Parameters
    ----------
    coco_json: path to a COCO instances JSON file or a pre-loaded dict
    out_labels_dir: output directory where one .txt per image is written
    out_classes_path: output path for classes.txt
    keep_category_ids: if True, YOLO indices equal COCO category ids (may be sparse)
    skip_empty_labels: if True, do not write .txt files for images without annotations
    """

    if isinstance(coco_json, Path):
        with coco_json.open("r", encoding="utf-8") as f:
            coco = json.load(f)
    else:
        coco = coco_json

    images = {img["id"]: img for img in coco.get("images", [])}
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    if not images:
        raise ValueError("No images found in COCO file.")
    if not cats:
        # Synthesize names if missing
        cat_ids = sorted({ann["category_id"] for ann in anns})
        cats = [{"id": cid, "name": f"category_{cid}"} for cid in cat_ids]

    # Decide YOLO class indexing
    if keep_category_ids:
        # YOLO class indices == COCO category ids (may be sparse)
        catid_to_yolo = {c["id"]: c["id"] for c in cats}
        yolo_classes: List[Optional[str]] = [None] * (max(catid_to_yolo.values()) + 1)
        for c in cats:
            idx = c["id"]
            name = c.get("name") or f"category_{idx}"
            yolo_classes[idx] = name
        for i, v in enumerate(yolo_classes):
            if v is None:
                yolo_classes[i] = f"category_{i}"
    else:
        # Remap to dense 0..K-1 in ascending order of COCO cat id
        sorted_cats = sorted(cats, key=lambda c: c["id"])
        catid_to_yolo = {c["id"]: i for i, c in enumerate(sorted_cats)}
        yolo_classes = [c.get("name") or f"category_{c['id']}" for c in sorted_cats]

    ensure_dir(out_labels_dir)

    # Group annotations by image
    anns_by_image: Dict[int, List[dict]] = {}
    for a in anns:
        if a.get("iscrowd", 0) == 1:
            continue
        img_id = a["image_id"]
        if img_id not in images:
            continue
        anns_by_image.setdefault(img_id, []).append(a)

    # Write label files
    for img_id, imginfo in images.items():
        w, h = imginfo.get("width"), imginfo.get("height")
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            raise ValueError(
                f"Image {imginfo.get('file_name')} missing valid width/height in COCO."
            )

        label_lines: List[str] = []
        for a in anns_by_image.get(img_id, []):
            cat_id = a["category_id"]
            if cat_id not in catid_to_yolo:
                continue
            yolo_cls = catid_to_yolo[cat_id]
            x, y, bw, bh = a["bbox"]
            # COCO xywh (absolute pixels) -> YOLO normalized (xc,yc,w,h)
            xc = (x + bw / 2.0) / w
            yc = (y + bh / 2.0) / h
            ww = bw / w
            hh = bh / h

            def clamp01(v: float) -> float:
                return max(0.0, min(1.0, v))

            xc = clamp01(xc)
            yc = clamp01(yc)
            ww = clamp01(ww)
            hh = clamp01(hh)

            label_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        stem = Path(imginfo["file_name"]).stem
        if label_lines or not skip_empty_labels:
            with (out_labels_dir / f"{stem}.txt").open("w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

    # Write classes.txt
    with out_classes_path.open("w", encoding="utf-8") as f:
        for name in yolo_classes:
            f.write(f"{name}\n")
