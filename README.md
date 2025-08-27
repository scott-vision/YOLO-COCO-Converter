# YOLO-COCO-Converter

Tools for working with COCO detection datasets. The primary utility is a
merger that combines multiple COCO `instances` JSON files into a single
file while handling id conflicts and optional file name prefixing.

## Features
- Remaps image, annotation and license identifiers to avoid collisions
- Validates categories across datasets
- Optional alignment of categories by name when ids differ
- Optional prefixing of image `file_name` fields or dropping duplicates
- Keeps `info` from the first dataset and merges/deduplicates licenses

## Command line usage
```bash
python -m coco_merge.cli \
  --inputs path/to/ds1.json path/to/ds2.json \
  --out merged.json
```

Common flags:
- `--prefix-mode` – one of `none`, `basename` or `custom`
- `--custom-prefixes` – custom prefixes when `--prefix-mode=custom`
- `--align-by-name` – align categories by name instead of id
- `--drop-duplicate-filenames` – skip images that would duplicate file names

## As a library
```python
from pathlib import Path
from coco_merge import merge_datasets

merged = merge_datasets([Path('a.json'), Path('b.json')], prefix_mode='basename')
```
Write the result with `json.dump` or similar.

## License
This project is released under the MIT license. See [LICENSE](LICENSE).
