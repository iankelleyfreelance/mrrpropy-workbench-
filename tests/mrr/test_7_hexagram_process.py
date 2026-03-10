import pytest
from datetime import datetime
import numpy as np
from pathlib import Path
import xarray as xr

import matplotlib

matplotlib.use("Agg")  # imprescindible en CI/headless

from mrrpropy.hexagram import plot_process_to_hexagram, PROCESS_SIGNATURES

OUTPUT_DIR = Path(r"./tests/figures/processes_in_hexagram")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_plot_hexagram_process():
    """Test básico para plot_hexagram_process."""   
    for process_ in PROCESS_SIGNATURES.keys(): 
        fig, filepath = plot_process_to_hexagram(process=process_, 
                                                 k=11,
                                                 tol_center=0.15,
                                                #  valid_threshold=0.5,
                                                 savefig=True,                                                 
                                                 output_dir=OUTPUT_DIR)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert filepath.exists()