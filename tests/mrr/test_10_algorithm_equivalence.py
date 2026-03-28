from __future__ import annotations

import numpy as np

import mrrpropy.RaProMPro_original as rpm_original
import mrrpropy.RaProMPro_optimized as rpm_optimized


def _assert_arrays_equal_with_nan(left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    assert np.array_equal(np.isnan(left), np.isnan(right))
    mask = np.isfinite(left) & np.isfinite(right)
    if np.any(mask):
        assert np.allclose(left[mask], right[mask], atol=0.0, rtol=0.0)


def test_mrrpronoise2_equivalence_cases() -> None:
    cases = [
        np.array([np.nan, np.nan, np.nan, np.nan], dtype=float),
        np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        np.array([0.2, 0.3, 0.25, 5.0, 0.22, 0.21], dtype=float),
        np.array([0.05, 0.08, 0.07, 0.06, 1.8, 0.09, 0.07], dtype=float),
    ]

    for vector in cases:
        out_original, noise_original = rpm_original.MrrProNoise2(
            vector.copy(), 0, 100.0, 60
        )
        out_optimized, noise_optimized = rpm_optimized.MrrProNoise2(
            vector.copy(), 0, 100.0, 60
        )

        _assert_arrays_equal_with_nan(
            np.asarray(out_original, dtype=float),
            np.asarray(out_optimized, dtype=float),
        )
        if np.isnan(noise_original) and np.isnan(noise_optimized):
            continue
        assert noise_original == noise_optimized


def test_continuity_equivalence_low_delta_h() -> None:
    vector_original = [np.nan, 1.0, 2.0, np.nan, 5.0, np.nan]
    matrix_original = [
        np.array([1.0, 1.0], dtype=float),
        np.array([2.0, 2.0], dtype=float),
        np.array([3.0, 3.0], dtype=float),
        np.array([4.0, 4.0], dtype=float),
        np.array([5.0, 5.0], dtype=float),
        np.array([6.0, 6.0], dtype=float),
    ]

    vector_optimized = vector_original.copy()
    matrix_optimized = [row.copy() for row in matrix_original]

    out_vector_original, out_matrix_original = rpm_original.Continuity(
        vector_original, matrix_original, 50.0
    )
    out_vector_optimized, out_matrix_optimized = rpm_optimized.Continuity(
        vector_optimized, matrix_optimized, 50.0
    )

    _assert_arrays_equal_with_nan(
        np.asarray(out_vector_original, dtype=float),
        np.asarray(out_vector_optimized, dtype=float),
    )
    for left, right in zip(out_matrix_original, out_matrix_optimized, strict=True):
        _assert_arrays_equal_with_nan(
            np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        )


def test_continuity_equivalence_high_delta_h() -> None:
    vector_original = [np.nan, 1.0, np.nan, 4.0, np.nan]
    matrix_original = [
        np.array([1.0, 1.0, 1.0], dtype=float),
        np.array([2.0, 2.0, 2.0], dtype=float),
        np.array([3.0, 3.0, 3.0], dtype=float),
        np.array([4.0, 4.0, 4.0], dtype=float),
        np.array([5.0, 5.0, 5.0], dtype=float),
    ]

    vector_optimized = vector_original.copy()
    matrix_optimized = [row.copy() for row in matrix_original]

    out_vector_original, out_matrix_original = rpm_original.Continuity(
        vector_original, matrix_original, 150.0
    )
    out_vector_optimized, out_matrix_optimized = rpm_optimized.Continuity(
        vector_optimized, matrix_optimized, 150.0
    )

    _assert_arrays_equal_with_nan(
        np.asarray(out_vector_original, dtype=float),
        np.asarray(out_vector_optimized, dtype=float),
    )
    for left, right in zip(out_matrix_original, out_matrix_optimized, strict=True):
        _assert_arrays_equal_with_nan(
            np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        )


def test_process_equivalence_small_synthetic_case() -> None:
    rpm_original.Nbins = 4
    rpm_original.NbinsM = 4
    rpm_original.fNy = 1.0
    rpm_original.lamb = 0.012
    rpm_original.K2w = 0.92
    rpm_original.SigmaScatt = np.ones((4, 4), dtype=float)
    rpm_original.SigmaExt = np.ones((4, 4), dtype=float)
    rpm_original.speed = np.arange(0.0, 4.0, 1.0)

    rpm_optimized.Nbins = 4
    rpm_optimized.NbinsM = 4
    rpm_optimized.fNy = 1.0
    rpm_optimized.lamb = 0.012
    rpm_optimized.K2w = 0.92
    rpm_optimized.SigmaScatt = np.ones((4, 4), dtype=float)
    rpm_optimized.SigmaExt = np.ones((4, 4), dtype=float)
    rpm_optimized.speed = np.arange(0.0, 4.0, 1.0)

    matrix = np.array(
        [
            [0.1, 0.2, 0.3, 0.1],
            [0.2, 0.1, 0.25, 0.05],
            [0.15, 0.05, 0.2, 0.1],
            [0.12, 0.18, 0.22, 0.08],
        ],
        dtype=float,
    )
    he = np.array([100.0, 200.0, 300.0, 400.0], dtype=float)
    D = np.array(
        [
            [0.5, 0.8, 1.2, 1.5],
            [0.5, 0.8, 1.2, 1.5],
            [0.5, 0.8, 1.2, 1.5],
            [0.5, 0.8, 1.2, 1.5],
        ],
        dtype=float,
    )
    neta = np.array([0.01, 0.02, 0.015, 0.018], dtype=float)
    noi_spe_ref = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

    out_original = rpm_original.Process(
        matrix.copy(),
        he.copy(),
        0.0,
        D.copy(),
        1.0,
        neta.copy(),
        1.0,
        0,
        noi_spe_ref.copy(),
    )
    out_optimized = rpm_optimized.Process(
        matrix.copy(),
        he.copy(),
        0.0,
        D.copy(),
        1.0,
        neta.copy(),
        1.0,
        0,
        noi_spe_ref.copy(),
    )

    assert len(out_original) == len(out_optimized)

    for left, right in zip(out_original, out_optimized, strict=True):
        left_arr = np.asarray(left, dtype=float)
        right_arr = np.asarray(right, dtype=float)
        _assert_arrays_equal_with_nan(left_arr, right_arr)
