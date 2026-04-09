from datetime import datetime

import matplotlib
import numpy as np
import pytest
import xarray as xr

from mrrpropy.analysis import processes as process_analysis

matplotlib.use("Agg")

pytestmark = [pytest.mark.integration]


def test_compute_layer_trend_ols(raprompro_subset_10min_loaded_mrr):
    result = raprompro_subset_10min_loaded_mrr.compute_layer_trend_ols(
        z_top=1000.0,
        z_base=2000.0,
        variable_threshold="Ze",
        threshold_value=-5.0,
        eps_mode="hourly_quantile",
        q=0.1,
        time_dim="time",
    )

    assert result.attrs["trend_method"] == "ols_legacy"
    for variable_name in ("Dm", "Nw", "LWC"):
        for prefix in ("b", "a", "r2", "F", "eps", "n_fit", "mask_fit"):
            assert f"{prefix}_{variable_name}" in result


def test_compute_layer_trend_kendall_theilsen(raprompro_subset_10min_loaded_mrr):
    result = raprompro_subset_10min_loaded_mrr.compute_layer_trend(
        z_top=1000.0,
        z_base=2000.0,
        variable_threshold="Ze",
        threshold_value=-5.0,
        time_dim="time",
        tau_zero_tol=0.05,
        min_points_trend=6,
    )

    assert result.attrs["trend_method"] == "kendall_theilsen"
    assert result.attrs["min_points_trend"] == 6
    assert "n_valid" in result

    n_time = raprompro_subset_10min_loaded_mrr.ds.sizes["time"]
    for variable_name in ("Dm", "Nw", "LWC"):
        for field_name in (
            f"tau_{variable_name}",
            f"p_{variable_name}",
            f"ts_{variable_name}",
            f"intercept_ts_{variable_name}",
            f"sign_{variable_name}",
            f"strength_{variable_name}",
            f"trend_mag_{variable_name}",
            f"trend_sign_{variable_name}",
            f"trend_strength_{variable_name}",
            f"trend_score_{variable_name}",
            f"trend_p_{variable_name}",
            f"n_fit_{variable_name}",
            f"mask_fit_{variable_name}",
            f"b_{variable_name}",
        ):
            assert field_name in result
        assert result[f"tau_{variable_name}"].shape == (n_time,)
        assert result[f"ts_{variable_name}"].shape == (n_time,)
        assert result[f"sign_{variable_name}"].dtype.kind in ("i", "u")


def test_compute_layer_trend_ols_exposes_canonical_trend_fields(
    raprompro_subset_10min_loaded_mrr,
):
    result = raprompro_subset_10min_loaded_mrr.compute_layer_trend(
        z_top=1000.0,
        z_base=2000.0,
        trend_method="ols",
        variable_threshold="Ze",
        threshold_value=-5.0,
        time_dim="time",
        min_points_trend=6,
    )

    assert result.attrs["trend_method"] == "ols"
    for variable_name in ("Dm", "Nw", "LWC"):
        for field_name in (
            f"b_{variable_name}",
            f"r2_{variable_name}",
            f"trend_mag_{variable_name}",
            f"trend_sign_{variable_name}",
            f"trend_strength_{variable_name}",
            f"trend_score_{variable_name}",
            f"trend_p_{variable_name}",
        ):
            assert field_name in result


def test_rain_process_analyze_uses_nonparametric_pipeline(
    raprompro_subset_10min_loaded_mrr,
    monkeypatch,
):
    def _raise_if_called(*args, **kwargs):
        raise AssertionError(
            "OLS helper should not be used by the default analysis pipeline."
        )

    monkeypatch.setattr(process_analysis, "ols_slope_intercept_r2", _raise_if_called)

    analysis = raprompro_subset_10min_loaded_mrr.rain_process_analyze(
        period=(datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 12, 10, 0)),
        layer=(1000.0, 2000.0),
        k=11,
        ze_th=-5.0,
        min_points_trend=6,
        tau_zero_tol=0.05,
        eps_q=0.01,
        rgb_q=0.02,
        vars_trend=("Dm", "Nw", "LWC"),
    )

    assert isinstance(analysis, xr.Dataset)
    assert "time" in analysis.coords
    assert analysis.sizes["time"] > 0

    for field_name in ("R", "G", "B", "minutes", "hex_x", "hex_y"):
        assert field_name in analysis

    for variable_name in ("Dm", "Nw", "LWC"):
        for prefix in (
            "tau",
            "p",
            "ts",
            "sign",
            "strength",
            "trend_mag",
            "trend_sign",
            "trend_strength",
            "trend_score",
            "trend_p",
        ):
            assert f"{prefix}_{variable_name}" in analysis
        assert f"b_{variable_name}" in analysis

    assert analysis.attrs["trend_method"] == "kendall_theilsen"
    assert analysis.attrs["rgb_method"] == "trend_score"
    assert analysis.attrs["min_points_trend"] == 6
    assert analysis.attrs["vars_trend"] == ("Dm", "Nw", "LWC")

    rgb_mapping = analysis.attrs.get("rgb_mapping", None)
    assert rgb_mapping == {"R": "Dm", "G": "Nw", "B": "LWC"}

    finite_r = analysis["R"].values[np.isfinite(analysis["R"].values)]
    if finite_r.size:
        assert np.nanmin(finite_r) >= 0.0
        assert np.nanmax(finite_r) <= 1.0

    classification = raprompro_subset_10min_loaded_mrr.classify_rain_process(
        analysis=analysis,
        min_tau_strength=0.10,
    )

    assert isinstance(classification, xr.Dataset)
    for field_name in (
        "proc_label",
        "sign_R",
        "sign_G",
        "sign_B",
        "strength",
        "tau_Dm",
        "tau_Nw",
        "tau_LWC",
        "trend_sign_Dm",
        "trend_strength_Dm",
        "trend_score_Dm",
    ):
        assert field_name in classification
    assert classification.attrs["classification_basis"] == "canonical_trend_sign"
