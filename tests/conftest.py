from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from mrrpropy.raw_class import MRRProData


RAW_DATA_PATH = Path(r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000.nc")
RAW_SUBSET_10MIN_PATH = Path(
    r"./tests/data/RAW_SUBSETS/mrrpro81/2025/03/08/20250308_120000_10min.nc"
)
RAPROMPRO_REFERENCE_PATH = Path(
    r"./tests/data/PRODUCTS/mrrpro81/2025/03/08/20250308_120000_raprompro.nc"
)
RAPROMPRO_REFERENCE_SUBSET_10MIN_PATH = Path(
    r"./tests/data/PRODUCT_SUBSETS/mrrpro81/2025/03/08/20250308_120000_raprompro_10min.nc"
)


def _sanitize_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "artifacts"


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


@pytest.fixture(scope="session")
def raw_dataset_path() -> Path:
    raw_path = Path(os.getenv("MRRPRO_RAW_DATA_PATH", str(RAW_DATA_PATH)))
    if not raw_path.exists():
        pytest.skip(f"Missing raw fixture file: {raw_path}")
    return raw_path


@pytest.fixture(scope="session")
def raw_subset_10min_path() -> Path:
    if not RAW_SUBSET_10MIN_PATH.exists():
        pytest.skip(f"Missing raw subset fixture file: {RAW_SUBSET_10MIN_PATH}")
    return RAW_SUBSET_10MIN_PATH


@pytest.fixture(scope="session")
def raprompro_reference_path() -> Path:
    product_path = Path(
        os.getenv("MRRPRO_REFERENCE_DATA_PATH", str(RAPROMPRO_REFERENCE_PATH))
    )
    if not product_path.exists():
        pytest.skip(f"Missing RaProMPro fixture file: {product_path}")
    return product_path


@pytest.fixture(scope="session")
def raprompro_reference_subset_10min_path() -> Path:
    if not RAPROMPRO_REFERENCE_SUBSET_10MIN_PATH.exists():
        pytest.skip(
            "Missing RaProMPro subset fixture file: "
            f"{RAPROMPRO_REFERENCE_SUBSET_10MIN_PATH}"
        )
    return RAPROMPRO_REFERENCE_SUBSET_10MIN_PATH


@pytest.fixture(scope="session")
def figure_root() -> Path:
    root = Path(os.getenv("MRRPRO_TEST_FIGURES", "./tests/figures")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def artifact_dir(figure_root: Path, request: pytest.FixtureRequest) -> Path:
    test_file = Path(str(request.node.fspath)).stem
    path = figure_root / _sanitize_path_part(test_file)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def generated_root() -> Path:
    root = Path(os.getenv("MRRPRO_TEST_GENERATED", "./tests/generated")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def generated_dir(generated_root: Path, request: pytest.FixtureRequest) -> Path:
    test_file = Path(str(request.node.fspath)).stem
    path = generated_root / _sanitize_path_part(test_file)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def raw_mrr(raw_dataset_path: Path) -> Iterator[MRRProData]:
    mrr = MRRProData.from_file(raw_dataset_path)
    yield mrr
    mrr.close()


@pytest.fixture(scope="session")
def raprompro_loaded_mrr(
    raw_dataset_path: Path,
    raprompro_reference_path: Path,
) -> Iterator[MRRProData]:
    mrr = MRRProData.from_file(raw_dataset_path)
    mrr.load_raprompro(raprompro_reference_path)
    yield mrr
    mrr.close()


@pytest.fixture(scope="session")
def raw_subset_10min_mrr(raw_subset_10min_path: Path) -> Iterator[MRRProData]:
    mrr = MRRProData.from_file(raw_subset_10min_path)
    yield mrr
    mrr.close()


@pytest.fixture(scope="session")
def raprompro_subset_10min_loaded_mrr(
    raw_subset_10min_path: Path,
    raprompro_reference_subset_10min_path: Path,
) -> Iterator[MRRProData]:
    mrr = MRRProData.from_file(raw_subset_10min_path)
    mrr.load_raprompro(raprompro_reference_subset_10min_path)
    yield mrr
    mrr.close()


@pytest.fixture(scope="session")
def generated_raprompro_path(
    raw_dataset_path: Path,
    generated_root: Path,
) -> Path:
    output_dir = generated_root / "generated_raprompro"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{raw_dataset_path.stem}_raprompro.nc"
    force_reprocess = _env_truthy("MRRPRO_FORCE_REPROCESS")

    if not force_reprocess and output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    mrr = MRRProData.from_file(raw_dataset_path)
    try:
        out = mrr.process_raprompro(
            save_dsd_3d=True,
            save_spe_3d=True,
            save=True,
            output_dir=output_dir,
        )
        out.close()
    finally:
        mrr.close()

    if not output_path.exists():
        raise FileNotFoundError(
            f"Expected generated file was not created: {output_path}"
        )
    return output_path
