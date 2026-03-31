from datetime import datetime

import matplotlib
import numpy as np
import pytest
import xarray as xr

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

    assert "b_Dm" in result
    assert "b_LWC" in result
    assert "b_Nw" in result
    assert "a_Dm" in result
    assert "a_LWC" in result
    assert "a_Nw" in result
    assert "r2_Dm" in result
    assert "r2_LWC" in result
    assert "r2_Nw" in result
    assert "F_Dm" in result
    assert "F_LWC" in result
    assert "F_Nw" in result
    assert result["b_Dm"].shape == (raprompro_subset_10min_loaded_mrr.ds.sizes["time"],)
    assert result["a_Dm"].shape == (raprompro_subset_10min_loaded_mrr.ds.sizes["time"],)
    assert result["r2_Dm"].shape == (
        raprompro_subset_10min_loaded_mrr.ds.sizes["time"],
    )
    assert result["F_Dm"].shape == (raprompro_subset_10min_loaded_mrr.ds.sizes["time"],)
    assert result["b_LWC"].shape == (
        raprompro_subset_10min_loaded_mrr.ds.sizes["time"],
    )
    assert result["a_LWC"].shape == (
        raprompro_subset_10min_loaded_mrr.ds.sizes["time"],
    )
    assert result["r2_LWC"].shape == (
        raprompro_subset_10min_loaded_mrr.ds.sizes["time"],
    )


def test_rain_process_analyze(raprompro_subset_10min_loaded_mrr):
    analysis = raprompro_subset_10min_loaded_mrr.rain_process_analyze(
        period=(datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 12, 10, 0)),
        layer=(1000.0, 2000.0),
        k=11,
        ze_th=-5.0,
        min_points_ols=10,
        eps_q=0.01,
        rgb_q=0.02,
        vars_trend=("Dm", "Nw", "LWC"),
    )

    assert isinstance(analysis, xr.Dataset)
    assert "time" in analysis.coords
    assert analysis.sizes["time"] > 0

    for v in ("R", "G", "B", "minutes", "hex_x", "hex_y"):
        assert v in analysis, f"Falta '{v}' en la salida de rain_process_analyze."

    for v in ("b_Dm", "b_Nw", "b_LWC"):
        assert v in analysis, f"Falta '{v}' en la salida (trends)."

    minutes = analysis["minutes"].values.astype(float)
    assert np.isfinite(minutes).any()
    assert np.nanmin(minutes) == pytest.approx(0.0)

    assert analysis.attrs.get("k", None) == 11
    assert analysis.attrs.get("z_top", None) == 1000.0
    assert analysis.attrs.get("z_base", None) == 2000.0
    assert analysis.attrs.get("eps_q", None) == 0.01
    assert analysis.attrs.get("rgb_q", None) == 0.02
    assert analysis.attrs.get("vars_trend", None) == ("Dm", "Nw", "LWC")

    rgb_mapping = analysis.attrs.get("rgb_mapping", None)
    assert rgb_mapping is not None
    assert rgb_mapping == {"R": "Dm", "G": "Nw", "B": "LWC"}

    t_out0 = np.datetime64(analysis["time"].values[0])
    t_out1 = np.datetime64(analysis["time"].values[-1])
    assert t_out0 >= np.datetime64("2025-03-08T12:00:00") - np.timedelta64(1, "s")
    assert t_out1 <= np.datetime64("2025-03-08T12:10:00") + np.timedelta64(1, "s")

    classification = raprompro_subset_10min_loaded_mrr.classify_rain_process(
        analysis=analysis
    )

    assert isinstance(classification, xr.Dataset)
    for v in ("proc_label", "sign_R", "sign_G", "sign_B", "strength"):
        assert v in classification
