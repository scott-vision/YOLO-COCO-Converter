"""Utilities for merging COCO detection datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Sequence


def load_json(p: Path) -> dict:
    """Load a JSON file as a Python dictionary."""
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def categories_signature(cats: List[dict]) -> List[Tuple[int, str]]:
    """Return a sortable signature for a list of categories."""
    return sorted(
        [(int(c["id"]), str(c.get("name", ""))) for c in cats],
        key=lambda x: x[0],
    )


def categories_name_map(cats: List[dict]) -> Dict[str, int]:
    """Map category name to id."""
    return {str(c.get("name", "")): int(c["id"]) for c in cats}


def dedup_licenses(all_licenses: List[dict]) -> Tuple[List[dict], Dict[Tuple[str, str], int]]:
    """Deduplicate licenses by ``(name, url)`` and assign contiguous ids."""
    key_to_newid: Dict[Tuple[str, str], int] = {}
    out = []
    next_id = 1
    for lic in all_licenses:
        name = str(lic.get("name", ""))
        url = str(lic.get("url", ""))
        key = (name, url)
        if key not in key_to_newid:
            key_to_newid[key] = next_id
            new_lic = {"id": next_id, "name": name}
            if url:
                new_lic["url"] = url
            out.append(new_lic)
            next_id += 1
    return out, key_to_newid


def merge_datasets(
    input_paths: Sequence[Path],
    *,
    prefix_mode: str = "none",
    custom_prefixes: Sequence[str] | None = None,
    align_by_name: bool = False,
    drop_duplicate_filenames: bool = False,
) -> Dict[str, Any]:
    """Merge multiple COCO datasets.

    Parameters
    ----------
    input_paths:
        Paths to the input COCO ``instances`` JSON files.
    prefix_mode:
        How to prefix image ``file_name`` values. ``"none"`` to keep names as-is,
        ``"basename"`` to prefix with the input file's stem, or ``"custom"`` to use
        ``custom_prefixes``.
    custom_prefixes:
        Custom prefixes to use when ``prefix_mode="custom"``.
    align_by_name:
        Align category ids by category name rather than raw id comparison.
    drop_duplicate_filenames:
        Drop images that would lead to duplicate ``file_name`` entries after
        prefixing.

    Returns
    -------
    dict
        The merged COCO dataset dictionary.
    """

    if prefix_mode == "custom":
        if not custom_prefixes or len(custom_prefixes) != len(input_paths):
            raise ValueError(
                "With prefix_mode='custom', custom_prefixes must match number of inputs"
            )

    datasets = [load_json(Path(p)) for p in input_paths]

    # Categories handling
    first_cats = datasets[0].get("categories", [])
    if not first_cats:
        raise ValueError("First dataset has no 'categories'.")
    first_sig = categories_signature(first_cats)
    first_name_to_id = categories_name_map(first_cats)

    cat_maps: List[Dict[int, int]] = []
    for i, ds in enumerate(datasets):
        cats = ds.get("categories", [])
        if not cats:
            raise ValueError(f"Dataset {input_paths[i]} has no 'categories'.")
        sig = categories_signature(cats)
        if sig != first_sig:
            if not align_by_name:
                raise ValueError(
                    f"Category mismatch between dataset 0 and dataset {i}. "
                    "Use align_by_name if names match but IDs differ."
                )
            # Align by name
            name_to_id_i = categories_name_map(cats)
            if set(name_to_id_i.keys()) != set(first_name_to_id.keys()):
                raise ValueError(
                    f"Cannot align categories for dataset {i}: names differ.\n"
                    f"First names: {sorted(first_name_to_id.keys())}\n"
                    f"Dataset {i} names: {sorted(name_to_id_i.keys())}"
                )
            cat_map = {name_to_id_i[name]: first_name_to_id[name] for name in name_to_id_i}
        else:
            cat_map = {int(c["id"]): int(c["id"]) for c in cats}
        cat_maps.append(cat_map)

    merged_categories = sorted(
        [{"id": int(c["id"]), "name": c.get("name", "")} for c in first_cats],
        key=lambda c: c["id"],
    )

    # Licenses: collect, dedup, and build mapping per dataset
    all_licenses = []
    for ds in datasets:
        all_licenses.extend(ds.get("licenses", []) or [])
    merged_licenses, lic_key_to_newid = dedup_licenses(all_licenses)

    def license_new_id(lic: dict) -> int | None:
        key = (str(lic.get("name", "")), str(lic.get("url", "")))
        return lic_key_to_newid.get(key, None)

    next_img_id = 1
    next_ann_id = 1

    merged_images: List[dict] = []
    merged_annotations: List[dict] = []

    seen_filenames: Set[str] = set()

    for i, (ds, inp_path) in enumerate(zip(datasets, input_paths)):
        prefix = ""
        if prefix_mode == "basename":
            prefix = Path(inp_path).stem + "_"
        elif prefix_mode == "custom":
            prefix = custom_prefixes[i]

        lic_map: Dict[int, int] = {}
        for lic in ds.get("licenses", []) or []:
            new_id = license_new_id(lic)
            if new_id is not None and "id" in lic:
                lic_map[int(lic["id"])] = new_id

        oldimg_to_newimg: Dict[int, int] = {}

        images = ds.get("images", []) or []
        anns = ds.get("annotations", []) or []

        anns_by_image: Dict[int, List[dict]] = {}
        for a in anns:
            anns_by_image.setdefault(int(a["image_id"]), []).append(a)

        for img in images:
            old_img_id = int(img["id"])
            file_name = str(img.get("file_name", ""))
            new_file_name = prefix + file_name if prefix else file_name

            if drop_duplicate_filenames and new_file_name in seen_filenames:
                continue

            seen_filenames.add(new_file_name)

            new_img = dict(img)
            new_img["id"] = next_img_id
            new_img["file_name"] = new_file_name
            if "license" in new_img and isinstance(new_img["license"], int):
                old_lic = int(new_img["license"])
                if old_lic in lic_map:
                    new_img["license"] = lic_map[old_lic]
                else:
                    new_img.pop("license", None)

            merged_images.append(new_img)
            oldimg_to_newimg[old_img_id] = next_img_id
            next_img_id += 1

            for a in anns_by_image.get(old_img_id, []):
                new_ann = dict(a)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = oldimg_to_newimg[old_img_id]
                old_cat = int(a["category_id"])
                new_ann["category_id"] = cat_maps[i].get(old_cat, old_cat)
                merged_annotations.append(new_ann)
                next_ann_id += 1

    merged = {
        "info": datasets[0].get("info", {"description": "Merged COCO dataset"}),
        "licenses": merged_licenses,
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories,
    }

    return merged
