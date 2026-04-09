from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow, pytest.mark.plot, pytest.mark.integration]


@pytest.fixture(scope="session")
def analysis(raprompro_subset_10min_loaded_mrr):
    return raprompro_subset_10min_loaded_mrr.rain_process_analyze(
        period=(datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 12, 10, 0)),
        layer=(1000.0, 2000.0),
        k=11,
        ze_th=-5.0,
        min_points_trend=10,
        eps_q=0.01,
        rgb_q=0.02,
        vars_trend=("Dm", "Nw", "LWC"),
    )


@pytest.fixture(scope="session")
def classified(raprompro_subset_10min_loaded_mrr, analysis):
    return raprompro_subset_10min_loaded_mrr.classify_rain_process(analysis=analysis)


def test_plot_rain_process_in_layer_2d(raprompro_subset_10min_loaded_mrr, artifact_dir):
    fig, path = raprompro_subset_10min_loaded_mrr.plot_rain_process_in_layer_2D(
        target_datetime=(
            datetime(2025, 3, 8, 12, 0, 0),
            datetime(2025, 3, 8, 12, 10, 0),
        ),
        layer=(1000.0, 2000.0),
        x="Dm",
        y="Nw",
        z="LWC",
        savefig=True,
        marker_size=100,
        figsize=(12, 10),
        cmap="seismic",
        output_dir=artifact_dir,
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    assert path.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")
    plt.close(fig)


def test_plot_rain_process_in_layer_hexagram(
    raprompro_subset_10min_loaded_mrr, analysis, artifact_dir
):
    (
        fig,
        filepath,
    ) = raprompro_subset_10min_loaded_mrr.plot_rain_process_in_layer_hexagram(
        analysis=analysis,
        savefig=True,
        output_dir=artifact_dir,
        dpi=200,
        alpha_hexagram=0.5,
        cmap="viridis",
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")

    axes = fig.get_axes()
    assert len(axes) >= 1
    ax0 = axes[0]
    assert isinstance(ax0, Axes)
    assert len(ax0.images) >= 1
    assert len(ax0.collections) >= 1
    plt.close(fig)


def test_plot_microphysics_summary_multipanel(
    raprompro_subset_10min_loaded_mrr,
    analysis,
    classified,
    artifact_dir,
):
    fig, path = raprompro_subset_10min_loaded_mrr.plot_processes_evolution(
        classified=classified,
        analysis=analysis,
        savefig=True,
        output_dir=artifact_dir,
        figsize=(14, 10),
        cmap="viridis",
        alpha_hexagram=0.5,
        markersize=40.0,
        line_width=0.8,
        dpi=200,
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    plt.close(fig)
