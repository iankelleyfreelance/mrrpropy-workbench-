from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mrrpropy.raw_class import MRRProData

pytestmark = [pytest.mark.slow, pytest.mark.integration]

KEY_VARIABLES = (
    "Type",
    "W",
    "spectral width",
    "DBPIA",
    "LWC",
    "RR",
    "Za",
    "Zea",
    "Ze",
    "SNR",
    "Nw",
    "Dm",
)


def test_raprompro_original_vs_optimized_equivalence(raw_dataset_path: Path) -> None:
    mrr = MRRProData.from_file(raw_dataset_path)
    try:
        ds_original = mrr.process_raprompro_original()
        ds_optimized = mrr.process_raprompro_optimized()

        assert tuple(ds_original.dims) == tuple(ds_optimized.dims)
        assert set(ds_original.data_vars) == set(ds_optimized.data_vars)
        assert np.array_equal(ds_original["time"].values, ds_optimized["time"].values)
        assert np.array_equal(ds_original["range"].values, ds_optimized["range"].values)

        for name in KEY_VARIABLES:
            assert name in ds_original, f"Missing {name} in original output"
            assert name in ds_optimized, f"Missing {name} in optimized output"

            left = ds_original[name].values.astype(float)
            right = ds_optimized[name].values.astype(float)

            assert left.shape == right.shape, f"Shape mismatch for {name}"
            assert np.array_equal(
                np.isfinite(left), np.isfinite(right)
            ), f"Finite-mask mismatch for {name}"

            mask = np.isfinite(left) & np.isfinite(right)
            if not np.any(mask):
                continue

            diff = np.abs(left[mask] - right[mask])
            assert float(np.nanmax(diff)) <= 1e-10, f"Max abs diff too large for {name}"
            assert float(np.nanmean(diff)) <= 1e-12, f"Mean abs diff too large for {name}"
    finally:
        mrr.close()
