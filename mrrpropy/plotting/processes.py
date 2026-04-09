from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import xarray as xr

from mrrpropy.hexagram import (
    PROCESS_CODES,
    PROCESS_MARKERS,
    PROCESS_SIGNATURES,
    get_hexagram_assets,
    get_process_hexagram_mask,
)


class SupportsProcessPlotting(Protocol):
    path: str | Path
    raprompro: xr.Dataset | None
    plot_cfg: Any

    def _is_processed(self) -> bool: ...


def plot_rain_process_in_layer_2d(
    subject: SupportsProcessPlotting,
    target_datetime: datetime | tuple[datetime, datetime],
    layer: tuple[float, float],
    x: str = "Dm",
    y: str = "LwC",
    z: str = "Nw",
    use_relative_difference: bool = True,
    savefig: bool = False,
    **kwargs: Any,
) -> tuple[Figure, Path | None]:
    """Plot the rain-process evolution in a selected layer as a 2D scatter."""
    if subject._is_processed():
        ds = subject.raprompro
    else:
        raise RuntimeError("Dataset is not processed.")
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")

    pcfg = subject.plot_cfg
    figsize = kwargs.get("figsize", pcfg.figsize)
    markersize = kwargs.get("marker_size", kwargs.get("markersize", 50))
    alpha = kwargs.get("alpha", 0.9)
    edgecolors = kwargs.get("edgecolors", "black")
    linewidths = kwargs.get("linewidths", 0.35)

    for var in (x, y, z):
        if var not in ds:
            raise KeyError(f"Variable '{var}' not found in dataset.")

    if isinstance(target_datetime, tuple) and target_datetime[0] >= target_datetime[1]:
        raise ValueError(
            "target_datetime tuple must be in increasing order (start, end)."
        )

    if isinstance(target_datetime, datetime):
        data = ds.sel(time=target_datetime, method="nearest").sel(range=slice(*layer))
    else:
        data = ds.sel(time=slice(*target_datetime)).sel(range=slice(*layer))

    last_range = data.range[-1]
    if use_relative_difference:
        diff = 100 * (data - data.sel(range=last_range)) / data.sel(
            range=slice(*layer)
        ).mean("range")
        data = diff.copy()
    else:
        data = data - data.sel(range=last_range)

    x_abs_max = np.abs(data[x].values[np.isfinite(data[x].values)]).max()
    y_abs_max = np.abs(data[y].values[np.isfinite(data[y].values)]).max()
    z_abs_max = np.abs(data[z].values[np.isfinite(data[z].values)]).max()

    fig, ax = plt.subplots(figsize=figsize)
    scatter_plot = cast(Any, data.plot.scatter)
    scatter_plot(
        x=x,
        y=y,
        hue=z,
        cmap=kwargs.get("cmap", "viridis"),
        s=markersize,
        vmin=-z_abs_max,
        vmax=z_abs_max,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths,
        ax=ax,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    figure = ax.get_figure()
    if figure is None:
        raise RuntimeError("Scatter plot figure could not be resolved.")
    colorbar_axis = figure.get_axes()[1]
    colorbar_axis.set_ylabel(z)

    zmin, zmax = layer
    if isinstance(target_datetime, tuple):
        ax.set_title(
            f"Layer {zmin/1000.}-{zmax/1000.} km \n {target_datetime[0]} to {target_datetime[1]}"
        )
    else:
        ax.set_title(f"Layer {zmin/1000.}-{zmax/1000.} km | {target_datetime}")

    ax.set_xlim(-x_abs_max, x_abs_max)
    ax.set_ylim(-y_abs_max, y_abs_max)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    fig.tight_layout()

    output_path: Path | None = None
    if savefig:
        output_dir = Path(kwargs.get("output_dir", Path.cwd()))
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(target_datetime, tuple):
            datestr = (
                f"{target_datetime[0].strftime('%Y%m%d_%H%M%S')}_to_"
                f"{target_datetime[1].strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            datestr = target_datetime.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / (
            f"rain_process_2D_{Path(subject.path).stem}_{datestr}_{zmin}-{zmax}m.png"
        )
        fig.savefig(output_path)

    return fig, output_path


def plot_rain_process_in_layer_hexagram(
    subject: SupportsProcessPlotting,
    *,
    analysis: xr.Dataset,
    use_snapped_colors: bool = True,
    savefig: bool = False,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Path | None]:
    """Overlay an analysed rain-process trajectory on the RGB hexagram."""
    pcfg = subject.plot_cfg
    figsize = kwargs.get("figsize", pcfg.figsize_multipanel)
    markersize = kwargs.get("markersize", pcfg.markersize)
    dpi = kwargs.get("dpi", pcfg.dpi)
    alpha = kwargs.get("alpha", pcfg.alpha_points)

    if analysis is None or not isinstance(analysis, xr.Dataset):
        raise TypeError("analysis debe ser un xr.Dataset (salida de rain_process_analyze).")

    required = ("hex_x", "hex_y", "minutes", "R", "G", "B")
    missing = [v for v in required if v not in analysis]
    if missing:
        raise KeyError(f"analysis no contiene variables requeridas: {missing}")

    k = analysis.attrs["k"]
    hex_assets = get_hexagram_assets(k=k)
    img = hex_assets["img"]
    ny, nx = img.shape[:2]

    hx = analysis["hex_x"].values.astype(float)
    hy = analysis["hex_y"].values.astype(float)
    minutes = analysis["minutes"].values.astype(float)

    if use_snapped_colors and all(v in analysis for v in ("snap_R", "snap_G", "snap_B")):
        colors = np.stack(
            [
                analysis["snap_R"].values,
                analysis["snap_G"].values,
                analysis["snap_B"].values,
            ],
            axis=1,
        ).astype(float)
    else:
        colors = np.stack(
            [analysis["R"].values, analysis["G"].values, analysis["B"].values],
            axis=1,
        ).astype(float)

    ok = (
        np.isfinite(hx)
        & np.isfinite(hy)
        & np.isfinite(minutes)
        & np.isfinite(colors).all(axis=1)
        & (hx >= 0)
        & (hy >= 0)
        & (hx <= (nx - 1))
        & (hy <= (ny - 1))
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        img,
        origin="lower",
        interpolation="nearest",
        alpha=kwargs.get("alpha_hexagram", 0.25),
    )

    scatter = None
    if np.any(ok):
        scatter = ax.scatter(
            hx[ok],
            hy[ok],
            s=markersize,
            c=minutes[ok],
            alpha=alpha,
            cmap=kwargs.get("cmap", "viridis"),
            edgecolors=kwargs.get("edgecolors", "black"),
            linewidths=kwargs.get("linewidths", 0.1),
        )

    z_top = analysis.attrs.get("z_top", None)
    z_base = analysis.attrs.get("z_base", None)
    t0s = analysis.attrs.get("period_start", None)
    t1s = analysis.attrs.get("period_end", None)
    layer_txt = (
        f"Capa {float(z_top):.0f}-{float(z_base):.0f} m"
        if (z_top is not None and z_base is not None)
        else "Capa (desconocida)"
    )
    period_txt = f"{t0s} → {t1s}" if (t0s is not None and t1s is not None) else ""
    ax.set_title(f"Hexagrama RGB (k={k}) | {layer_txt}\n{period_txt}".rstrip())
    ax.set_xlabel("hex_x (índice rejilla)")
    ax.set_ylabel("hex_y (índice rejilla)")
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.grid(False)

    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Minutes since start")

    fig.tight_layout()

    filepath: Path | None = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        safe_t0 = str(t0s or "t0").replace(":", "").replace("-", "").replace(" ", "_")
        safe_t1 = str(t1s or "t1").replace(":", "").replace("-", "").replace(" ", "_")
        safe_layer = (
            f"{float(z_top):.0f}-{float(z_base):.0f}m"
            if (z_top is not None and z_base is not None)
            else "layer"
        )
        filepath = outdir / f"rain_process_hex_{safe_t0}_{safe_t1}_{safe_layer}_k{k}.png"
        fig.savefig(filepath, dpi=dpi)

    return fig, filepath


def plot_processes_evolution(
    subject: SupportsProcessPlotting,
    *,
    classified: xr.Dataset,
    analysis: xr.Dataset | None = None,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Path | None]:
    """Plot a temporal summary of classified rain-process evolution."""
    pcfg = subject.plot_cfg
    cmap = kwargs.get("cmap", pcfg.cmap)
    figsize = kwargs.get("figsize", getattr(pcfg, "figsize_multipanel", (10, 5)))
    dpi = kwargs.get("dpi", pcfg.dpi)

    markersize_timeline = kwargs.get("markersize_timeline", 28.0)
    heatmap_cmap = kwargs.get("heatmap_cmap", "coolwarm")
    title_fs = kwargs.get("title_fs", 18)
    label_fs = kwargs.get("label_fs", 16)
    tick_fs = kwargs.get("tick_fs", 14)

    process_order = kwargs.get(
        "process_order",
        [
            "unknown",
            "steady_or_weak",
            "evaporation",
            "breakup",
            "coalescence",
            "growth",
            "activation",
            "autoconversion",
            "no_data",
        ],
    )

    if not isinstance(classified, xr.Dataset):
        raise TypeError("classified debe ser un xr.Dataset.")
    if "time" not in classified.coords:
        raise KeyError("classified debe tener coord 'time'.")
    for variable in ("proc_label", "strength", "sign_R", "sign_G", "sign_B"):
        if variable not in classified:
            raise KeyError(f"classified debe contener '{variable}'.")

    if analysis is not None:
        if not isinstance(analysis, xr.Dataset):
            raise TypeError("analysis debe ser xr.Dataset si se proporciona.")
        classified, analysis = xr.align(classified, analysis, join="inner")

    time_values = classified["time"].values
    if time_values.size == 0:
        raise ValueError("classified no contiene tiempos tras el alineamiento.")

    row_labels = ["R", "G", "B"]
    rgb_map = analysis.attrs.get("rgb_mapping", None) if analysis is not None else classified.attrs.get("rgb_mapping", None)
    if isinstance(rgb_map, dict) and all(k in rgb_map for k in ("R", "G", "B")):
        row_labels = [
            f"R ({rgb_map['R']})",
            f"G ({rgb_map['G']})",
            f"B ({rgb_map['B']})",
        ]

    df = classified[["proc_label", "strength"]].to_dataframe().reset_index()
    labels = df["proc_label"].astype(str).fillna("unknown").to_numpy()
    present_labels = list(pd.unique(labels))
    ordered_labels = [label for label in process_order if label in present_labels]
    extra_labels = [label for label in present_labels if label not in ordered_labels]
    unique_labels = ordered_labels + extra_labels
    if len(unique_labels) > 26:
        raise ValueError("Hay más de 26 categorías de proceso; A,B,C... ya no basta.")

    map_code = {label: PROCESS_CODES.get(label, label.upper()) for label in unique_labels}
    y_index = np.array([unique_labels.index(label) for label in labels], dtype=int)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[40, 1.2],
        height_ratios=[1.35, 1.0],
        wspace=0.08,
        hspace=0.50,
    )

    ax_timeline = fig.add_subplot(gs[0, 0])
    cax_timeline = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, 0], sharex=ax_timeline)
    cax_heatmap = fig.add_subplot(gs[1, 1])
    plt.setp(ax_timeline.get_xticklabels(), visible=False)

    scatter = ax_timeline.scatter(
        df["time"].to_numpy(),
        y_index,
        c=df["strength"].to_numpy(dtype=float),
        cmap=cmap,
        s=markersize_timeline,
        vmin=0.0,
        vmax=1.0,
        edgecolors="none",
    )
    ax_timeline.set_title("(a) Process timeline (color = strength)", fontsize=title_fs)
    ax_timeline.set_ylabel("Process", fontsize=label_fs)
    ax_timeline.set_yticks(range(len(unique_labels)))
    ax_timeline.set_yticklabels([map_code[label] for label in unique_labels], fontsize=tick_fs)
    ax_timeline.tick_params(labelsize=tick_fs)

    cb1 = fig.colorbar(scatter, cax=cax_timeline)
    cb1.set_label("strength (0–1)", fontsize=label_fs)
    cax_timeline.tick_params(labelsize=tick_fs)

    sign_r = classified["sign_R"].values.astype(float)
    sign_g = classified["sign_G"].values.astype(float)
    sign_b = classified["sign_B"].values.astype(float)
    signs = np.vstack([sign_r, sign_g, sign_b])
    signs = np.where(np.isfinite(signs), signs, 0.0)

    time_numeric = mdates.date2num(pd.to_datetime(time_values).to_pydatetime())
    if time_numeric.size == 1:
        delta = 1.0 / (24 * 60)
        time_numeric = np.array([time_numeric[0] - delta, time_numeric[0] + delta])

    heatmap = ax_heatmap.imshow(
        signs,
        aspect="auto",
        interpolation="nearest",
        cmap=plt.get_cmap(heatmap_cmap, 3),
        vmin=-1,
        vmax=1,
        extent=(time_numeric[0], time_numeric[-1], -0.5, 2.5),
        origin="lower",
    )
    ax_heatmap.set_title("(b) Signs heatmap (-1 / 0 / +1)", fontsize=title_fs)
    ax_heatmap.set_yticks([0, 1, 2])
    ax_heatmap.set_yticklabels(row_labels, fontsize=tick_fs)
    ax_heatmap.set_ylabel("RGB component", fontsize=label_fs)
    ax_heatmap.tick_params(labelsize=tick_fs)

    cb2 = fig.colorbar(heatmap, cax=cax_heatmap)
    cb2.set_label("sign", fontsize=label_fs)
    cb2.set_ticks([-1, 0, 1])
    cb2.set_ticklabels(["-1", "0", "+1"])
    cax_heatmap.tick_params(labelsize=tick_fs)

    ax_heatmap.xaxis_date()
    ax_heatmap.set_xlabel("Time", fontsize=label_fs)
    ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    filepath: Path | None = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        t0s = str(np.datetime_as_string(time_values[0], unit="s")).replace(":", "")
        t1s = str(np.datetime_as_string(time_values[-1], unit="s")).replace(":", "")
        if analysis is not None:
            z_top = analysis.attrs.get("z_top", "zt")
            z_base = analysis.attrs.get("z_base", "zb")
        else:
            z_top = classified.attrs.get("z_top", "zt")
            z_base = classified.attrs.get("z_base", "zb")
        filepath = outdir / f"processes_evolution_{z_top}-{z_base}_{t0s}-{t1s}.png"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    return fig, filepath


def plot_classified_processes_on_hexagram(
    subject: SupportsProcessPlotting,
    *,
    classified: xr.Dataset,
    analysis: xr.Dataset | None = None,
    show_background: bool = False,
    show_process_masks: bool = True,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Path | None]:
    """Plot classified samples on the package RGB hexagram."""
    pcfg = subject.plot_cfg
    figsize = kwargs.get("figsize", pcfg.figsize_hex)
    dpi = kwargs.get("dpi", pcfg.dpi)
    alpha_hexagram = kwargs.get("alpha_hexagram", pcfg.alpha_hexagram)
    alpha_mask = kwargs.get("alpha_mask", 0.18)
    markersize = kwargs.get("markersize", 35.0)

    if not isinstance(classified, xr.Dataset):
        raise TypeError("classified must be an xr.Dataset.")
    for variable in ("proc_label", "hex_x", "hex_y"):
        if variable not in classified:
            raise KeyError(f"classified must contain '{variable}'.")

    k = classified.attrs.get("k", None)
    if k is None and analysis is not None:
        k = analysis.attrs.get("k", None)
    if k is None:
        raise KeyError("k not found in classified.attrs nor analysis.attrs.")

    hex_assets = get_hexagram_assets(k=k)
    img = np.asarray(hex_assets["img"], float)

    process_colors = kwargs.get(
        "process_colors",
        {
            "breakup": "#d95f02",
            "coalescence": "#1b9e77",
            "evaporation": "#7570b3",
            "growth": "#e7298a",
            "activation": "#66a61e",
            "steady_or_weak": "#bdbdbd",
            "unknown": "#666666",
            "no_data": "#d9d9d9",
        },
    )

    fig, ax = plt.subplots(figsize=figsize)
    if show_background:
        ax.imshow(
            img,
            origin="lower",
            interpolation="nearest",
            alpha=alpha_hexagram,
        )

    if show_process_masks:
        for process in PROCESS_SIGNATURES:
            mask2d, _ = get_process_hexagram_mask(
                process,
                k=k,
                tol_center=classified.attrs["tol_center"],
            )
            color = process_colors.get(process, "#000000")
            rgba = np.zeros((*mask2d.shape, 4), dtype=float)
            rgb = plt.matplotlib.colors.to_rgb(color)
            rgba[mask2d, 0] = rgb[0]
            rgba[mask2d, 1] = rgb[1]
            rgba[mask2d, 2] = rgb[2]
            rgba[mask2d, 3] = alpha_mask
            ax.imshow(rgba, origin="lower", interpolation="nearest")

    hx = classified["hex_x"].values.astype(float)
    hy = classified["hex_y"].values.astype(float)
    labels = classified["proc_label"].values.astype(str)
    valid = np.isfinite(hx) & np.isfinite(hy) & (hx >= 0) & (hy >= 0)

    present_labels = pd.unique(labels[valid])
    handles: list[Any] = []
    strength = classified["strength"].values.astype(float)
    scatter = None
    for label in present_labels:
        mask = valid & (labels == label)
        if not np.any(mask):
            continue
        marker = PROCESS_MARKERS.get(label, "o")
        scatter = ax.scatter(
            hx[mask],
            hy[mask],
            s=markersize,
            c=strength[mask],
            cmap=kwargs.get("cmap", "viridis"),
            vmin=0.0,
            vmax=1.0,
            marker=marker,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.95,
            label=PROCESS_CODES.get(label, label.upper()),
        )
        handles.append(scatter)

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("strength")

    ax.set_title(f"Classified rain processes on RGB hexagram (k={k})")
    ax.set_xlabel("hex_x")
    ax.set_ylabel("hex_y")
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(-0.5, img.shape[0] - 0.5)
    ax.set_aspect("equal")
    ax.grid(False)

    if handles:
        ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()

    filepath: Path | None = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        t0s = str(np.datetime_as_string(classified["time"].values[0], unit="s")).replace(":", "")
        t1s = str(np.datetime_as_string(classified["time"].values[-1], unit="s")).replace(":", "")
        filepath = outdir / f"classified_processes_hexagram_{t0s}_{t1s}_k{k}.png"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    return fig, filepath
