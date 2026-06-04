from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import pytest

from mrrpropy.plotting.processes import plot_fused_process_quicklook

from tests.rain_processes.test_process_scan_plots import (
    MIN_TAU_STRENGTH,
    WINDOW_STEP_M,
    WINDOW_THICKNESS_M,
    _scan_artifact_dir,
)


matplotlib.use("Agg")


@pytest.fixture(scope="session")
def scan_df(raprompro_subset_10min_loaded_mrr) -> pd.DataFrame:
    return raprompro_subset_10min_loaded_mrr.build_column_process_scan_dataframe(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
    )


def _make_fused_df_from_scan_snapshot(scan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a minimal fused_df-like dataframe from a single-time scan snapshot.

    This keeps plotting tests fast while still relying on the same scan_df data
    produced by the scan workflow tests.
    """
    if scan_df.empty:
        return pd.DataFrame()

    time0 = pd.Timestamp(pd.to_datetime(scan_df["time"]).iloc[0])
    snap = scan_df[scan_df["time"] == time0].copy()
    if snap.empty:
        return pd.DataFrame()

    if "window_id" in snap.columns:
        snap = snap.sort_values("window_id", ascending=False)
    take = snap.head(3)
    if take.empty:
        return pd.DataFrame()

    label = str(take["proc_label"].astype(str).iloc[0])
    z_top = float(pd.to_numeric(take["z_top_m"], errors="coerce").max())
    z_bottom = float(pd.to_numeric(take["z_bottom_m"], errors="coerce").min())

    return pd.DataFrame(
        {
            "time": [time0],
            "proc_label_fused": [label],
            "z_top_fused": [z_top],
            "z_bottom_fused": [z_bottom],
        }
    )


@pytest.fixture(scope="session")
def fused_df(scan_df) -> pd.DataFrame:
    return _make_fused_df_from_scan_snapshot(scan_df)


def test_plot_fused_process_quicklook_savefig(scan_df, fused_df, artifact_dir: Path):
    output_dir = _scan_artifact_dir(artifact_dir)

    fig, path = plot_fused_process_quicklook(
        scan_df,
        fused_df,
        processes=[
            "breakup",
            "growth_depletion",
            "growth_depletion_loss",
            "growth_depletion_gain",
            "activation",
            "evaporation",
            "growth",
        ],
        savefig=True,
        output_dir=output_dir,
        dpi=200,
    )
    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    axes = fig.get_axes()
    assert len(axes) == 1
    assert len(axes[0].patches) >= 1
    plt.close(fig)


def test_plot_fused_process_quicklook_returns_none_path_by_default(scan_df, fused_df):
    fig, path = plot_fused_process_quicklook(scan_df, fused_df)
    assert isinstance(fig, Figure)
    assert path is None
    plt.close(fig)


def test_plot_fused_process_quicklook_processes_filter(scan_df, fused_df):
    if fused_df.empty:
        pytest.skip("No fused events available for filter test.")

    first_label = str(pd.unique(fused_df["proc_label_fused"].astype(str))[0])

    fig_keep, _ = plot_fused_process_quicklook(
        scan_df, fused_df, processes=[first_label]
    )
    assert len(fig_keep.axes[0].patches) >= 1
    plt.close(fig_keep)

    fig_drop, _ = plot_fused_process_quicklook(
        scan_df, fused_df, processes=["__no_such_process__"]
    )
    assert len(fig_drop.axes[0].patches) == 0
    plt.close(fig_drop)
