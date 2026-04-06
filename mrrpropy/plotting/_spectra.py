from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import xarray as xr


class SupportsSpectralAccess(Protocol):
    path: str | Path
    ds: xr.Dataset
    raprompro: xr.Dataset | None


def nearest_period(
    subject: SupportsSpectralAccess,
    target_datetime: datetime | np.datetime64,
    target_range: float,
) -> tuple[np.datetime64, float]:
    """Return the nearest available time and range values."""
    ds = subject.ds
    t_sel = cast(
        np.datetime64,
        ds["time"].sel(time=target_datetime, method="nearest").values,
    )
    r_sel = float(ds["range"].sel(range=target_range, method="nearest").values)
    return t_sel, r_sel


def get_velocity_axis(subject: SupportsSpectralAccess, n_bins: int) -> np.ndarray:
    """
    Build a Doppler velocity axis when the dataset does not expose one directly.

    MRR-PRO files often omit an explicit per-bin velocity coordinate. When that
    happens, infer it from the VEL fold limit if available, otherwise fall back
    to a 12 m/s Nyquist-like upper bound.
    """
    ds = subject.ds
    vny = 12.0
    if "VEL" in ds and isinstance(ds["VEL"].attrs, dict):
        if "fold_limit_upper" in ds["VEL"].attrs:
            try:
                vny = float(ds["VEL"].attrs["fold_limit_upper"])
            except Exception:
                pass
    return np.linspace(0.0, vny, int(n_bins), dtype=float)


def get_spectrum_1d(
    subject: SupportsSpectralAccess,
    target_datetime: datetime | np.datetime64,
    target_range: float,
    *,
    spectrum_var: str = "spectrum_reflectivity",
) -> tuple[np.datetime64, float, np.ndarray, np.ndarray, str]:
    """
    Extract the nearest 1D spectrum at a requested time and range.

    Supports either a dense cube ``(time, range, spectrum_n_samples)`` or the
    indexed MRR-PRO layout using ``index_spectra(time, range)``.
    """
    ds = subject.ds
    if spectrum_var not in ds:
        if "spectrum_raw" in ds:
            spectrum_var = "spectrum_raw"
        else:
            raise KeyError(
                f"No encuentro '{spectrum_var}' ni 'spectrum_raw' en el Dataset."
            )

    t_sel, r_sel = nearest_period(subject, target_datetime, target_range)

    da = ds[spectrum_var]
    units = str(da.attrs.get("units", ""))
    bin_dim = "spectrum_n_samples"
    if bin_dim not in da.dims:
        raise ValueError(
            f"'{spectrum_var}' no tiene dimensiÃ³n '{bin_dim}'. dims={da.dims}"
        )

    if ("time" in da.dims) and ("range" in da.dims):
        spectrum = da.sel(time=t_sel, range=r_sel, method="nearest").values.astype(float)
        vel = get_velocity_axis(subject, spectrum.shape[-1])
        return t_sel, r_sel, vel, spectrum, units

    if ("time" in da.dims) and ("n_spectra" in da.dims):
        if "index_spectra" not in ds:
            raise KeyError(
                f"'{spectrum_var}' es (time,n_spectra,bin) pero falta 'index_spectra(time,range)'."
            )
        idx = (
            ds["index_spectra"]
            .sel(time=t_sel, range=r_sel, method="nearest")
            .values
        )
        js = int(idx)
        spectrum = da.sel(time=t_sel, n_spectra=js).values.astype(float)
        vel = get_velocity_axis(subject, spectrum.shape[-1])
        return t_sel, r_sel, vel, spectrum, units

    raise ValueError(f"Formato de '{spectrum_var}' no soportado. dims={da.dims}")


def get_spectrogram_2d(
    subject: SupportsSpectralAccess,
    target_datetime: datetime | np.datetime64,
    *,
    spectrum_var: str,
    range_limits: tuple[float, float] | None = None,
) -> tuple[np.datetime64, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Extract a 2D range-by-spectrum view for the nearest requested time.
    """
    ds: xr.Dataset = subject.ds

    if spectrum_var not in ds:
        if subject.raprompro is None or spectrum_var not in subject.raprompro:
            raise KeyError(f"'{spectrum_var}' not found.")
        ds = subject.raprompro

    da = ds[spectrum_var]

    t_sel = cast(
        np.datetime64,
        da["time"].sel(time=target_datetime, method="nearest").values,
    )
    units = str(da.attrs.get("units", "?"))

    if range_limits is None:
        r0 = float(ds["range"].min().values)
        r1 = float(ds["range"].max().values)
    else:
        r0, r1 = map(float, range_limits)

    ranges = ds["range"].sel(range=slice(r0, r1)).values.astype(float)

    if "spectrum_n_samples" in da.sizes:
        n_bins = da.sizes["spectrum_n_samples"]
        vel = get_velocity_axis(subject, int(n_bins))
    else:
        if subject.raprompro is not None and "speed" in subject.raprompro:
            vel = np.asarray(subject.raprompro["speed"].values, dtype=float)
        else:
            raise ValueError("velocity not found in raprompro Dataset.")

    if ("time" in da.dims) and ("range" in da.dims):
        spec2d = da.sel(time=t_sel, range=slice(r0, r1)).values.astype(float)
        return t_sel, ranges, vel, spec2d, units

    if ("time" in da.dims) and ("n_spectra" in da.dims):
        if "index_spectra" not in ds:
            raise KeyError(
                f"'{spectrum_var}' es (time,n_spectra,bin) pero falta 'index_spectra(time,range)'."
            )
        idx_vec = (
            ds["index_spectra"].sel(time=t_sel, range=slice(r0, r1)).values.astype(int)
        )
        slab = da.sel(time=t_sel).values.astype(float)
        spec2d = slab[idx_vec, :]
        return t_sel, ranges, vel, spec2d, units

    raise ValueError(f"Formato de '{spectrum_var}' no soportado. dims={da.dims}")
