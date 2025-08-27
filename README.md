# YOLO-COCO-Converter

YOLO <-> COCO conversion tools with an extra COCO dataset merger. Use the unified
CLI for conversions and merging, or import functions in notebooks.

## Features
- YOLO -> COCO: Build COCO JSON from YOLO labels and image sizes
- COCO -> YOLO: Write YOLO .txt labels and `classes.txt` from COCO
- Merge COCO: Merge multiple COCO datasets with id remapping and options
- Optional Pillow for image size detection; or provide a sizes CSV

## Install (optional)
For a global CLI (Pillow is included by default):
```bash
pip install -e .
```
This installs `yolococo` (primary) and `coco-merge` (alias for merge-only).

## CLI Usage
Run as a module (no install required):
```bash
python -m yolococo ...
```

Or after installing:
```bash
yolococo ...
```

Subcommands
- YOLO -> COCO:
  ```bash
  yolococo yolo2coco \
    --images ./images \
    --labels ./labels \
    --classes ./classes.txt \    # optional
    --sizes ./sizes.csv \        # optional; overrides Pillow sizes
    --bbox-round 2 \             # decimals for bbox/area (use <0 to disable)
    --file-name-mode name \      # name | relative
    --out ./coco.json
  ```
  sizes.csv format (no header): `filename,width,height`.

- COCO -> YOLO:
  ```bash
  yolococo coco2yolo \
    --coco ./instances.json \
    --out-labels ./yolo_labels \
    --out-classes ./classes.txt \
    [--keep-category-ids]
    [--skip-empty-labels]
  ```

- Merge COCO:
  ```bash
  yolococo merge \
    --inputs path/to/ds1.json path/to/ds2.json \
    --out merged.json \
    [--prefix-mode none|basename|custom] \
    [--custom-prefixes A_ B_] \
    [--align-by-name] \
    [--drop-duplicate-filenames]
  ```

## Programmatic Use (incl. Jupyter)
```python
from pathlib import Path
import json
from yolococo import yolo_to_coco, coco_to_yolo_files, merge_datasets

# YOLO -> COCO
coco = yolo_to_coco(
    images_dir=Path("./images"),
    labels_dir=Path("./labels"),
    classes_path=Path("./classes.txt"),  # or None
    sizes_csv=None,  # or Path("./sizes.csv")
)
with open("coco.json", "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False, indent=2)

# COCO -> YOLO (writes files)
coco_to_yolo_files(Path("./instances.json"), Path("./yolo_labels"), Path("./classes.txt"))

# Merge COCO
merged = merge_datasets([Path("a.json"), Path("b.json")], prefix_mode="basename")
```
Tip for notebooks: run from the repo root (so `import yolococo` works), or add the repo path to `sys.path`.

## Notes & Assumptions
- YOLO labels follow the common format: `<cls> <xc> <yc> <w> <h>` normalized to [0,1].
- COCO expects absolute pixel `bbox` as `[x_min, y_min, width, height]`.
- Pillow is installed by default and used to read image sizes; `--sizes` CSV (if provided) takes precedence per matching filename.
- Bounding boxes and area are rounded to `--bbox-round` decimals (default 2). Set a negative value to disable rounding.
- Merge assumes consistent semantic classes across inputs; use `--align-by-name` if ids differ but names match.
- `--file-name-mode` controls whether COCO `images[].file_name` stores just the basename (`name`) or the path relative to `--images` (`relative`). When using `relative`, directory separators are `/`.

## License
MIT - see [LICENSE](LICENSE).

## Testing & Visualization
- Install dev deps: `pip install -e .[test]`
- Run tests: `pytest -q`
- Artifacts (COCO JSON and annotated images) are written to `tests/_artifacts/` for manual inspection.

Manual visualization script:
```bash
python scripts/visualize_labels.py
```
It converts the sample in `test/` to COCO and saves overlays: `sample_annotated_from_yolo.jpg`, `sample_annotated_from_coco.jpg`, and the original image in `tests/_artifacts/`.

