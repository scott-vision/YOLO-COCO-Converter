from __future__ import annotations

from pathlib import Path
import json
from typing import List, Tuple

from PIL import Image, ImageDraw

from yolococo import yolo_to_coco


def _read_yolo_labels_txt(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in (5, 6):
            continue
        cls = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        ww = float(parts[3])
        hh = float(parts[4])
        rows.append((cls, xc, yc, ww, hh))
    return rows


def _draw_boxes_on_image(img: Image.Image, boxes, color=(255, 0, 0), width=3, labels=None) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if labels:
            try:
                draw.text((x1 + 2, y1 + 2), labels[i], fill=color)
            except Exception:
                pass
    return out


def test_yolo_to_coco_and_visualize(images_dir: Path, labels_dir: Path, sample_image: Path, sample_label: Path, artifacts_dir: Path):
    # Convert YOLO -> COCO
    coco = yolo_to_coco(images_dir, labels_dir, classes_path=None, sizes_csv=None, bbox_round=2, file_name_mode="name")

    # Basic sanity checks
    assert "images" in coco and "annotations" in coco and "categories" in coco
    assert any(img["file_name"] == sample_image.name for img in coco["images"])  # image present

    # Save COCO JSON artifact for manual inspection
    (artifacts_dir / "coco.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    # Visualize YOLO labels on sample image
    img = Image.open(sample_image).convert("RGB")
    W, H = img.size
    yolo_rows = _read_yolo_labels_txt(sample_label)
    yolo_boxes = []
    for cls, xc, yc, ww, hh in yolo_rows:
        abs_w = ww * W
        abs_h = hh * H
        x_min = (xc * W) - (abs_w / 2.0)
        y_min = (yc * H) - (abs_h / 2.0)
        x1 = max(0.0, x_min)
        y1 = max(0.0, y_min)
        x2 = min(W, x1 + abs_w)
        y2 = min(H, y1 + abs_h)
        yolo_boxes.append((x1, y1, x2, y2))

    vis_yolo = _draw_boxes_on_image(img, yolo_boxes, color=(255, 0, 0))
    vis_yolo.save(artifacts_dir / "sample_annotated_from_yolo.jpg", quality=95)

    # Visualize COCO annotations for same image
    # Find image id
    img_rec = next(i for i in coco["images"] if i["file_name"] == sample_image.name)
    img_id = img_rec["id"]
    coco_boxes = []
    coco_labels = []
    for a in coco["annotations"]:
        if a["image_id"] != img_id:
            continue
        x, y, w, h = a["bbox"]
        coco_boxes.append((x, y, x + w, y + h))
        coco_labels.append(str(a.get("category_id")))

    # Basic consistency: same number of boxes between YOLO and COCO for this image
    assert len(coco_boxes) == len(yolo_boxes)

    vis_coco = _draw_boxes_on_image(img, coco_boxes, color=(0, 255, 0), labels=coco_labels)
    vis_coco.save(artifacts_dir / "sample_annotated_from_coco.jpg", quality=95)

    # Save original too for comparison
    img.save(artifacts_dir / "sample_original.jpg", quality=95)
