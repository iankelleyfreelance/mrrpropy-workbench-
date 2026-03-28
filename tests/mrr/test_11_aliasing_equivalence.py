from __future__ import annotations

import numpy as np

import mrrpropy.RaProMPro_optimized as rpm_optimized


def _assert_arrays_equal_with_nan(left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    assert np.array_equal(np.isnan(left), np.isnan(right))
    mask = np.isfinite(left) & np.isfinite(right)
    if np.any(mask):
        assert np.allclose(left[mask], right[mask], atol=0.0, rtol=0.0)


def test_group_equivalence_small_case() -> None:
    rpm_optimized.NbinsM = 4

    a = np.array(
        [np.nan, 0.2, 0.4, 0.1, np.nan, 0.3, 0.6, 0.2, np.nan, np.nan, np.nan, np.nan],
        dtype=float,
    )
    d = np.arange(a.size, dtype=float)

    out_original = rpm_optimized.group_original(a.copy(), 5, 3, d.copy())
    out_candidate = rpm_optimized.group_optimized(a.copy(), 5, 3, d.copy())

    assert len(out_original) == len(out_candidate)
    for left, right in zip(out_original, out_candidate, strict=True):
        _assert_arrays_equal_with_nan(
            np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        )


def test_aliasing_equivalence_small_case() -> None:
    rpm_optimized.Nbins = 4
    rpm_optimized.NbinsM = 4

    matrix = np.array(
        [
            [0.1, np.nan, 0.3, 0.1],
            [0.2, 0.1, np.nan, 0.05],
            [0.15, 0.05, 0.2, np.nan],
            [0.12, 0.18, 0.22, 0.08],
        ],
        dtype=float,
    )
    he = np.array([100.0, 200.0, 300.0, 400.0], dtype=float)

    out_original = rpm_optimized.Aliasing_original(matrix.copy(), 1.0, he.copy(), 0.0)
    out_candidate = rpm_optimized.Aliasing_optimized(matrix.copy(), 1.0, he.copy(), 0.0)

    assert len(out_original) == len(out_candidate)
    for left, right in zip(out_original, out_candidate, strict=True):
        _assert_arrays_equal_with_nan(
            np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        )
