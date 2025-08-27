from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

from yolococo import yolo_to_coco


def read_yolo_labels_txt(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in (5, 6):
            continue
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return rows


def draw_boxes(img: Image.Image, boxes, color=(255, 0, 0), width=3):
    out = img.copy()
    d = ImageDraw.Draw(out)
    for (x1, y1, x2, y2) in boxes:
        d.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return out


def main():
    repo_root = Path(__file__).resolve().parents[1]
    images_dir = repo_root / "test"
    labels_dir = repo_root / "test"
    artifacts = repo_root / "tests" / "_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # Convert YOLO -> COCO
    coco = yolo_to_coco(images_dir, labels_dir)
    (artifacts / "coco.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    # Visualize first image
    img_path = next(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    lbl_path = labels_dir / f"{img_path.stem}.txt"
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    rows = read_yolo_labels_txt(lbl_path) if lbl_path.exists() else []
    yboxes = []
    for cls, xc, yc, ww, hh in rows:
        abs_w = ww * W
        abs_h = hh * H
        x_min = (xc * W) - (abs_w / 2.0)
        y_min = (yc * H) - (abs_h / 2.0)
        x1 = max(0.0, x_min)
        y1 = max(0.0, y_min)
        x2 = min(W, x1 + abs_w)
        y2 = min(H, y1 + abs_h)
        yboxes.append((x1, y1, x2, y2))

    vis_yolo = draw_boxes(img, yboxes, color=(255, 0, 0))
    vis_yolo.save(artifacts / "sample_annotated_from_yolo.jpg", quality=95)

    # Visualize COCO
    img_rec = next(i for i in coco["images"] if i["file_name"] == img_path.name)
    img_id = img_rec["id"]
    cboxes = []
    for a in coco["annotations"]:
        if a["image_id"] != img_id:
            continue
        x, y, w, h = a["bbox"]
        cboxes.append((x, y, x + w, y + h))
    vis_coco = draw_boxes(img, cboxes, color=(0, 255, 0))
    vis_coco.save(artifacts / "sample_annotated_from_coco.jpg", quality=95)
    img.save(artifacts / "sample_original.jpg", quality=95)


if __name__ == "__main__":
    main()

