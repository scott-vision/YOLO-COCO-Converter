from __future__ import annotations

from pathlib import Path
import os
import shutil
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_image(repo_root: Path) -> Path:
    p = repo_root / "test" / "20220329_4_1_P1t001.jpg"
    assert p.exists(), f"Sample image not found: {p}"
    return p


@pytest.fixture(scope="session")
def sample_label(repo_root: Path) -> Path:
    p = repo_root / "test" / "20220329_4_1_P1t001.txt"
    assert p.exists(), f"Sample label not found: {p}"
    return p


@pytest.fixture(scope="session")
def images_dir(sample_image: Path) -> Path:
    return sample_image.parent


@pytest.fixture(scope="session")
def labels_dir(sample_label: Path) -> Path:
    return sample_label.parent


@pytest.fixture(scope="session")
def artifacts_dir(repo_root: Path) -> Path:
    out = repo_root / "tests" / "_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    return out

