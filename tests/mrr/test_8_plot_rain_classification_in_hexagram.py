import pytest
from datetime import datetime
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")  # imprescindible en CI/headless

from mrrpropy.raw_class import (
    MRRProData,
)  # cambia 'mrrpro' por el nombre real de tu módulo

# Ruta por defecto al fichero de prueba.
# Puedes sobrescribirla con la variable de entorno MRRPRO_TEST_FILE.
MRR_PATH = Path(r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000.nc")
RPP_PATH = Path(
    r"./tests/data/PRODUCTS/mrrpro81/2025/03/08/20250308_120000_raprompro.nc"
)
OUTPUT_DIR = Path(r"./tests/figures/mrr_plots_hexagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def mrr():
    """Carga una instancia de MRRProData para todos los tests."""
    if not MRR_PATH.exists():
        pytest.skip(f"No se encuentra el archivo de data: {MRR_PATH}")
    mrr = MRRProData.from_file(MRR_PATH)
    mrr.load_raprompro(RPP_PATH)
    yield mrr
    mrr.close()
    
@pytest.fixture(scope="session")
def analysis(mrr):
    """Salida de rain_process_analyze reutilizable (hexagrama)."""
    return mrr.rain_process_analyze(
        period=(datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 12, 31, 0)),
        layer=(1000.0, 2000.0),
        k=11,
        ze_th=-5.0,
        min_points_ols=10,
        eps_q=0.01,
        rgb_q=0.02,
        vars_trend=("Dm", "Nw", "LWC"),
    )

@pytest.fixture(scope="session")
def classified(mrr, analysis):
    """Clasificación reutilizable."""
    return mrr.classify_rain_process(analysis=analysis)

def test_plot_classified_processes_on_hexagram(mrr,analysis, classified):
    """Test básico: plot_microphysics_summary_multipanel (plot-only) debe ejecutar y guardar figura."""
    
    fig, path = mrr.plot_classified_processes_on_hexagram(
        classified=classified,
        analysis=analysis,        
        savefig=True,
        output_dir=OUTPUT_DIR,
        **{
            "figsize": (14, 10),
            "cmap": "viridis",
            "alpha_hexagram": 0.5,
            "markersize": 70.0,
            "line_width": 0.8,
            "dpi": 200,
            "legend_fontsize": 14,
            "alpha_hexagram": 0.25
            
        },
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()

    plt.close(fig)

