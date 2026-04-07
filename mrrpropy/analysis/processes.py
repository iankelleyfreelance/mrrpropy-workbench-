from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import xarray as xr

from mrrpropy.hexagram import (
    PROCESS_SIGNATURES,
    build_rgb_from_trends,
    get_hexagram_assets,
    map_rgb_to_hexagram,
)
from mrrpropy.utils import compute_eps, ols_slope_intercept_r2


class SupportsRainAnalysis(Protocol):
    path: str | Path
    raprompro: xr.Dataset | None

    def _is_processed(self) -> bool: ...


def compute_layer_trend_ols(
    subject: SupportsRainAnalysis,
    *,
    z_top: float,
    z_base: float,
    time_dim: str = "time",
    variable_threshold: str = "Ze",
    threshold_value: float = -5.0,
    vars: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    eps_mode: str = "hourly_quantile",
    q: float = 0.01,
    eps_floor_mode: str = "global_min",
    min_points_ols: int = 10,
) -> xr.Dataset:
    """Compute layer-wise OLS trends of selected microphysical variables."""
    if not subject._is_processed():
        raise RuntimeError("MRR-Pro data not processed (raprompro missing).")

    ds = subject.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")

    if z_base <= z_top:
        raise ValueError("z_base must be greater than z_top (in meters).")

    layer = ds.sel({"range": slice(z_top, z_base)})

    if time_dim not in layer.coords:
        raise KeyError(f"Missing coord '{time_dim}' in dataset.")
    if "range" not in layer.coords:
        raise KeyError("Missing coord 'range' in dataset.")
    if variable_threshold not in layer:
        raise KeyError(f"Missing threshold variable '{variable_threshold}' in dataset.")

    for variable_name in vars:
        if variable_name not in layer:
            raise KeyError(f"Missing variable '{variable_name}' in dataset.")

    z_layer = layer["range"].values.astype(float)
    depth = np.abs(z_layer - float(z_top)).astype(float)
    dz = float(z_base - z_top)

    ze = layer[variable_threshold]
    ze_mask = xr.where(np.isfinite(ze) & (ze > threshold_value), True, False)

    out = xr.Dataset(
        coords={
            time_dim: layer[time_dim].values,
            "range_layer": layer["range"].values,
        }
    )
    out = out.assign_coords(depth=("range_layer", depth))

    out["dz"] = xr.DataArray(dz)
    out["mask_ze"] = xr.DataArray(
        ze_mask.values.astype(bool),
        dims=(time_dim, "range_layer"),
    )

    out.attrs.update(
        dict(
            z_top=float(z_top),
            z_base=float(z_base),
            dz=float(dz),
            variable_threshold=str(variable_threshold),
            threshold_value=float(threshold_value),
            vars=tuple(vars),
            eps_mode=str(eps_mode),
            eps_floor_mode=str(eps_floor_mode),
            q=float(q),
            min_points_ols=int(min_points_ols),
        )
    )

    global_eps: dict[str, float] = {}
    if eps_mode == "global_quantile" or eps_floor_mode == "global_min":
        for variable_name in vars:
            global_eps[variable_name] = compute_eps(layer[variable_name].values, q=q)

    times = layer[time_dim].values
    ntime = times.size
    nrange = layer.sizes["range"]
    ze_mask_np = ze_mask.values.astype(bool)

    n_valid = np.sum(ze_mask_np, axis=1).astype(int)
    out["n_valid"] = xr.DataArray(n_valid, dims=(time_dim,))

    for variable_name in vars:
        b_arr = np.full(ntime, np.nan, dtype=float)
        a_arr = np.full(ntime, np.nan, dtype=float)
        r2_arr = np.full(ntime, np.nan, dtype=float)
        f_arr = np.full(ntime, np.nan, dtype=float)

        eps_used = np.full(ntime, np.nan, dtype=float)
        n_fit = np.zeros(ntime, dtype=int)
        mask_fit = np.zeros((ntime, nrange), dtype=bool)

        values = layer[variable_name].values.astype(float)

        for index in range(ntime):
            if n_valid[index] < min_points_ols:
                continue

            mask = (
                ze_mask_np[index, :]
                & np.isfinite(values[index, :])
                & (values[index, :] > 0.0)
            )
            nmask = int(np.sum(mask))
            if nmask < min_points_ols:
                continue

            if eps_mode == "hourly_quantile":
                eps_t = compute_eps(values[index, :], q=q)
            elif eps_mode == "global_quantile":
                eps_t = global_eps.get(variable_name, np.nan)
            else:
                raise ValueError(f"Unsupported eps_mode={eps_mode!r}")

            if not np.isfinite(eps_t) or eps_t <= 0:
                continue

            if eps_floor_mode == "global_min":
                eps_g = global_eps.get(variable_name, np.nan)
                if np.isfinite(eps_g) and eps_g > 0:
                    eps_t = max(float(eps_t), float(eps_g))

            x = depth[mask]
            y = np.log(np.maximum(values[index, mask], eps_t))

            b, a, r2 = ols_slope_intercept_r2(x, y)
            if not (np.isfinite(b) and np.isfinite(a) and np.isfinite(r2)):
                continue

            b_arr[index] = float(b)
            a_arr[index] = float(a)
            r2_arr[index] = float(r2)
            f_arr[index] = float(np.exp(b * dz))

            mask_fit[index, :] = mask
            n_fit[index] = nmask
            eps_used[index] = float(eps_t)

        out[f"b_{variable_name}"] = xr.DataArray(b_arr, dims=(time_dim,))
        out[f"a_{variable_name}"] = xr.DataArray(a_arr, dims=(time_dim,))
        out[f"r2_{variable_name}"] = xr.DataArray(r2_arr, dims=(time_dim,))
        out[f"F_{variable_name}"] = xr.DataArray(f_arr, dims=(time_dim,))

        out[f"eps_{variable_name}"] = xr.DataArray(eps_used, dims=(time_dim,))
        out[f"n_fit_{variable_name}"] = xr.DataArray(n_fit, dims=(time_dim,))
        out[f"mask_fit_{variable_name}"] = xr.DataArray(
            mask_fit,
            dims=(time_dim, "range_layer"),
        )

    return out


def rain_process_analyze(
    subject: SupportsRainAnalysis,
    *,
    period: tuple[datetime, datetime],
    layer: tuple[float, float],
    k: int,
    ze_th: float = -5.0,
    min_points_ols: int = 10,
    eps_q: float = 0.01,
    rgb_q: float = 0.02,
    vars_trend: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
) -> xr.Dataset:
    """Analyse rain-process evolution in a layer over a selected period."""
    if not subject._is_processed():
        raise RuntimeError("Dataset not preprocessed / raprompro not available.")

    ds = subject.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")

    z_top, z_base = layer
    if z_base <= z_top:
        raise ValueError("layer debe cumplir z_base > z_top (metros).")

    t0, t1 = period
    if t0 >= t1:
        raise ValueError("period debe ser creciente (t0 < t1).")

    ds_sub = ds.sel(time=slice(t0, t1))
    if ds_sub.sizes.get("time", 0) == 0:
        raise ValueError("Selección temporal vacía: revisa period.")

    trends = compute_layer_trend_ols(
        subject,
        z_top=z_top,
        z_base=z_base,
        variable_threshold="Ze",
        threshold_value=ze_th,
        vars=vars_trend,
        q=eps_q,
        min_points_ols=min_points_ols,
    )

    trends = trends.sel(time=slice(ds_sub["time"].values[0], ds_sub["time"].values[-1]))

    rgb = build_rgb_from_trends(
        trends,
        vars=(f"b_{vars_trend[0]}", f"b_{vars_trend[1]}", f"b_{vars_trend[2]}"),
        q=rgb_q,
    )

    time_values = rgb["time"].values
    time_start = time_values[0]
    minutes = ((time_values - time_start) / np.timedelta64(1, "m")).astype(float)

    hex_assets = get_hexagram_assets(k=k)
    hex_ds = map_rgb_to_hexagram(rgb, hex_assets=hex_assets)

    out = xr.Dataset(coords={"time": rgb["time"].values})

    for variable_name in trends.data_vars:
        out[variable_name] = trends[variable_name]
    for coord_name in trends.coords:
        if coord_name not in out.coords:
            out = out.assign_coords({coord_name: trends.coords[coord_name]})
    out.attrs.update(trends.attrs)

    out["R"] = rgb["R"]
    out["G"] = rgb["G"]
    out["B"] = rgb["B"]
    out["minutes"] = xr.DataArray(minutes, dims=("time",))

    for variable_name in hex_ds.data_vars:
        out[variable_name] = hex_ds[variable_name]

    out.attrs.update(
        dict(
            period_start=str(np.datetime_as_string(ds_sub["time"].values[0], unit="s")),
            period_end=str(np.datetime_as_string(ds_sub["time"].values[-1], unit="s")),
            z_top=float(z_top),
            z_base=float(z_base),
            k=int(k),
            ze_th=float(ze_th),
            min_points_ols=int(min_points_ols),
            eps_q=float(eps_q),
            rgb_q=float(rgb_q),
            vars_trend=tuple(vars_trend),
            rgb_convention=str(
                f"R={vars_trend[0]}, G={vars_trend[1]}, B={vars_trend[2]}"
            ),
        )
    )
    out.attrs["rgb_mapping"] = {
        "R": vars_trend[0],
        "G": vars_trend[1],
        "B": vars_trend[2],
    }
    out.attrs["strength_definition"] = "min(|RGB-0.5|)/0.5"
    return out


def classify_rain_process(
    subject: SupportsRainAnalysis,
    *,
    analysis: xr.Dataset,
    tol_center: float = 0.05,
    min_strength: float = 0.10,
) -> xr.Dataset:
    """Classify each time sample into a rain-process category."""
    del subject

    if analysis is None or not isinstance(analysis, xr.Dataset):
        raise TypeError("analysis debe ser un xr.Dataset (salida de rain_process_analyze).")
    if "time" not in analysis.coords:
        raise KeyError("analysis debe tener coord 'time'.")
    for variable_name in ("R", "G", "B"):
        if variable_name not in analysis:
            raise KeyError(
                "analysis debe incluir R,G,B (decisión tomada en rain_process_analyze)."
            )

    expected = {"R": "Dm", "G": "Nw", "B": "LWC"}
    rgb_map = analysis.attrs.get("rgb_mapping", None)
    if rgb_map != expected:
        raise ValueError(f"rgb_mapping={rgb_map} pero este clasificador espera {expected}.")

    r_values = analysis["R"].values.astype(float)
    g_values = analysis["G"].values.astype(float)
    b_values = analysis["B"].values.astype(float)
    ok = np.isfinite(r_values) & np.isfinite(g_values) & np.isfinite(b_values)

    def _sign_from_center(values: np.ndarray, tol: float) -> np.ndarray:
        sign = np.zeros(values.shape, dtype=int)
        sign[values > 0.5 + tol] = +1
        sign[values < 0.5 - tol] = -1
        return sign

    def _strength(values: np.ndarray) -> np.ndarray:
        return np.clip(np.abs(values - 0.5) / 0.5, 0.0, 1.0)

    sign_r = np.zeros(r_values.shape, dtype=int)
    sign_g = np.zeros(g_values.shape, dtype=int)
    sign_b = np.zeros(b_values.shape, dtype=int)
    if np.any(ok):
        sign_r[ok] = _sign_from_center(r_values[ok], tol_center)
        sign_g[ok] = _sign_from_center(g_values[ok], tol_center)
        sign_b[ok] = _sign_from_center(b_values[ok], tol_center)

    strength = np.zeros(r_values.shape, dtype=float)
    if np.any(ok):
        strength[ok] = np.minimum.reduce(
            [
                _strength(r_values[ok]),
                _strength(g_values[ok]),
                _strength(b_values[ok]),
            ]
        )

    def _normalize_signatures(signature_definition: Any) -> list[tuple[int, int, int]]:
        if isinstance(signature_definition, tuple) and len(signature_definition) == 3:
            return [
                (
                    int(signature_definition[0]),
                    int(signature_definition[1]),
                    int(signature_definition[2]),
                )
            ]
        if isinstance(signature_definition, (list, tuple)):
            signatures: list[tuple[int, int, int]] = []
            for item in signature_definition:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    signatures.append((int(item[0]), int(item[1]), int(item[2])))
            if signatures:
                return signatures
        raise ValueError(f"Firma no válida: {signature_definition!r}")

    label = np.full(r_values.shape, "no_data", dtype=object)
    label[ok] = "unknown"

    for process_name, signature_definition in PROCESS_SIGNATURES.items():
        signatures = _normalize_signatures(signature_definition)
        process_mask = np.zeros(r_values.shape, dtype=bool)
        for sign_r_expected, sign_g_expected, sign_b_expected in signatures:
            process_mask |= (
                ok
                & (sign_r == sign_r_expected)
                & (sign_g == sign_g_expected)
                & (sign_b == sign_b_expected)
            )
        take = process_mask & (label == "unknown")
        label[take] = process_name

    weak = ok & (strength < min_strength)
    label[weak] = "steady_or_weak"

    out = xr.Dataset(coords={"time": analysis["time"].values})
    out["proc_label"] = xr.DataArray(label, dims=("time",))
    out["sign_R"] = xr.DataArray(sign_r, dims=("time",))
    out["sign_G"] = xr.DataArray(sign_g, dims=("time",))
    out["sign_B"] = xr.DataArray(sign_b, dims=("time",))
    out["strength"] = xr.DataArray(strength, dims=("time",))

    out["R"] = analysis["R"]
    out["G"] = analysis["G"]
    out["B"] = analysis["B"]

    for variable_name in (
        "hex_x",
        "hex_y",
        "hex_area",
        "minutes",
        "snap_R",
        "snap_G",
        "snap_B",
    ):
        if variable_name in analysis:
            out[variable_name] = analysis[variable_name]

    out.attrs["tol_center"] = float(tol_center)
    out.attrs["min_strength"] = float(min_strength)
    out.attrs["rgb_mapping"] = rgb_map

    for key in (
        "rgb_convention",
        "period_start",
        "period_end",
        "z_top",
        "z_base",
        "k",
        "rgb_q",
        "eps_q",
        "ze_th",
        "min_points_ols",
    ):
        if key in analysis.attrs:
            out.attrs[key] = analysis.attrs[key]

    return out
