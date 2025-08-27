"""YOLO <-> COCO conversion tools.

Primary API:
- yolo_to_coco: build a COCO dict from YOLO labels and images
- coco_to_yolo_files: write YOLO labels and classes.txt from a COCO dataset
- merge_datasets: merge multiple COCO datasets (re-export)
"""

from .convert import yolo_to_coco, coco_to_yolo_files
from coco_merge.merger import merge_datasets

__version__ = "0.2.0"
__all__ = ["__version__", "yolo_to_coco", "coco_to_yolo_files", "merge_datasets"]

