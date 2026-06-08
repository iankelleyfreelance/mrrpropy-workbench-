from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")


DEFAULT_INPUT_FILE = Path(
    r"workbench/output/raprompro/2025/10/29/20251029_190000_raprompro.nc"
)
DEFAULT_OUTPUT_DIR = Path(r"workbench/output/histograms")
DEFAULT_VARIABLE = "Dm"


def _parse_timestamp_or_none(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _slugify_timestamp(value: pd.Timestamp | None, fallback: str) -> str:
    if value is None:
        return fallback
    return value.strftime("%Y%m%d_%H%M%S")


def _format_height_label(height_min: float | None, height_max: float | None) -> str:
    if height_min is None and height_max is None:
        return "all_heights"
    if height_min is None:
        return f"top_to_{int(height_max)}m"
    if height_max is None:
        return f"{int(height_min)}m_to_top"
    return f"{int(height_min)}m_to_{int(height_max)}m"


def _select_dataset(
    ds: xr.Dataset,
    *,
    variable_name: str,
    time_start: pd.Timestamp | None,
    time_end: pd.Timestamp | None,
    height_min: float | None,
    height_max: float | None,
    all_heights: bool,
) -> xr.DataArray:
    if variable_name not in ds:
        raise KeyError(f"Variable '{variable_name}' not found in dataset.")
    if "time" not in ds.coords:
        raise KeyError("The dataset does not contain a 'time' coordinate.")
    if "range" not in ds.coords:
        raise KeyError("The dataset does not contain a 'range' coordinate.")

    data = ds[variable_name]

    if time_start is not None or time_end is not None:
        if time_start is None:
            time_start = pd.Timestamp(ds["time"].values[0])
        if time_end is None:
            time_end = pd.Timestamp(ds["time"].values[-1])
        if time_end < time_start:
            raise ValueError("--time-end must be greater than or equal to --time-start.")
        data = data.sel(time=slice(time_start.to_datetime64(), time_end.to_datetime64()))

    if not all_heights:
        if height_min is not None and height_max is not None and height_max < height_min:
            raise ValueError(
                "--height-max must be greater than or equal to --height-min."
            )
        if height_min is not None or height_max is not None:
            range_values = ds["range"].values.astype(float)
            lower = float(range_values.min()) if height_min is None else float(height_min)
            upper = float(range_values.max()) if height_max is None else float(height_max)
            data = data.sel(range=slice(lower, upper))

    if data.size == 0:
        raise RuntimeError("The selected time/height window contains no samples.")

    return data


def _compute_bin_edges(
    values: np.ndarray,
    *,
    bins: int,
    bin_width: float | None,
    x_limits: tuple[float, float] | None,
) -> int | np.ndarray:
    if bin_width is None:
        return bins

    if bin_width <= 0:
        raise ValueError("--bin-width must be greater than 0.")

    if x_limits is not None:
        xmin, xmax = x_limits
    else:
        xmin = float(np.nanmin(values))
        xmax = float(np.nanmax(values))

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise RuntimeError("Could not determine finite bin edges from the selected data.")
    if xmax <= xmin:
        xmax = xmin + bin_width

    return np.arange(xmin, xmax + bin_width, bin_width, dtype=float)


def _summarize_values(values: np.ndarray) -> dict[str, float]:
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _format_variable_label(data: xr.DataArray, fallback_name: str) -> str:
    variable_name = str(data.name) if data.name is not None else fallback_name
    units = data.attrs.get("units")
    if isinstance(units, str) and units.strip():
        return f"{variable_name} ({units.strip()})"
    return variable_name


def _format_variable_title(data: xr.DataArray, fallback_name: str) -> str:
    for key in ("long_name", "description"):
        value = data.attrs.get(key)
        if isinstance(value, str) and value.strip():
            units = data.attrs.get("units")
            if isinstance(units, str) and units.strip():
                return f"{value.strip()} [{units.strip()}]"
            return value.strip()
    return _format_variable_label(data, fallback_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a histogram from a processed *_raprompro.nc file using a selected "
            "variable, time window, and height range."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the processed *_raprompro.nc file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the histogram PNG and summary CSV will be written.",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=DEFAULT_VARIABLE,
        help="Variable to plot. Default: Dm",
    )
    parser.add_argument(
        "--time-start",
        type=str,
        default=None,
        help="Optional start time, for example 2025-10-29T13:00:00.",
    )
    parser.add_argument(
        "--time-end",
        type=str,
        default=None,
        help="Optional end time, for example 2025-10-29T13:59:59.",
    )
    parser.add_argument(
        "--height-min",
        type=float,
        default=None,
        help="Optional lower bound of the height range in metres.",
    )
    parser.add_argument(
        "--height-max",
        type=float,
        default=None,
        help="Optional upper bound of the height range in metres.",
    )
    parser.add_argument(
        "--all-heights",
        action="store_true",
        help="Use all available heights and ignore --height-min/--height-max.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins when --bin-width is not provided.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help="Optional fixed histogram bin width.",
    )
    parser.add_argument(
        "--x-limits",
        nargs=2,
        type=float,
        default=None,
        help="Optional x-axis limits for the histogram.",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot probability density instead of raw counts.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Keep only values strictly greater than zero.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional legend label for the histogram.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom figure title.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(6.5, 6.5),
        help="Figure size in inches, for example --figsize 6.5 6.5.",
    )
    args = parser.parse_args()

    input_file = args.input_file.resolve()
    output_dir = args.output_dir.resolve()
    time_start = _parse_timestamp_or_none(args.time_start)
    time_end = _parse_timestamp_or_none(args.time_end)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if input_file.suffix.lower() != ".nc" or "_raprompro" not in input_file.stem:
        raise ValueError("--input-file must point to a processed *_raprompro.nc file.")
    if args.bins < 1:
        raise ValueError("--bins must be at least 1.")
    if args.all_heights and (args.height_min is not None or args.height_max is not None):
        raise ValueError("--all-heights cannot be combined with height limits.")
    if args.x_limits is not None and args.x_limits[1] <= args.x_limits[0]:
        raise ValueError("--x-limits must be increasing.")

    output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(input_file)
    try:
        selected = _select_dataset(
            ds,
            variable_name=args.variable,
            time_start=time_start,
            time_end=time_end,
            height_min=args.height_min,
            height_max=args.height_max,
            all_heights=args.all_heights,
        )

        values = np.asarray(selected.values, dtype=float).ravel()
        valid_mask = np.isfinite(values)
        if args.positive_only:
            valid_mask &= values > 0.0
        values = values[valid_mask]

        if values.size == 0:
            raise RuntimeError("No valid values remain after applying the selection.")

        x_label = _format_variable_label(selected, args.variable)
        variable_title = _format_variable_title(selected, args.variable)

        bin_edges = _compute_bin_edges(
            values,
            bins=args.bins,
            bin_width=args.bin_width,
            x_limits=tuple(args.x_limits) if args.x_limits is not None else None,
        )

        time_values = pd.to_datetime(selected["time"].values)
        time_label_start = time_values[0]
        time_label_end = time_values[-1]

        range_values = selected["range"].values.astype(float)
        height_label = _format_height_label(
            None if args.all_heights else args.height_min,
            None if args.all_heights else args.height_max,
        )

        fig, ax = plt.subplots(figsize=tuple(args.figsize), constrained_layout=True)
        ax.hist(
            values,
            bins=bin_edges,
            density=args.density,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.85,
            label=args.label,
        )

        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        ax.axvline(mean_value, color="#d62728", linewidth=1.8, linestyle="-", label="mean")
        ax.axvline(
            median_value,
            color="#2ca02c",
            linewidth=1.8,
            linestyle="--",
            label="median",
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Density" if args.density else "Count")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        if args.x_limits is not None:
            ax.set_xlim(tuple(args.x_limits))

        if args.title is None:
            ax.set_title(
                (
                    f"{variable_title} histogram | {input_file.stem}\n"
                    f"{time_label_start.strftime('%Y-%m-%d %H:%M:%S')} to "
                    f"{time_label_end.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"heights {range_values.min():.0f}-{range_values.max():.0f} m | "
                    f"n={values.size}"
                ),
                fontsize=12,
            )
        else:
            ax.set_title(args.title)

        if args.label is not None:
            ax.legend()
        else:
            ax.legend(loc="upper right")

        time_slug_start = _slugify_timestamp(time_start, "alltimes_start")
        time_slug_end = _slugify_timestamp(time_end, "alltimes_end")
        output_stem = (
            f"{input_file.stem}_{args.variable}_hist_"
            f"{time_slug_start}_{time_slug_end}_{height_label}"
        )

        figure_path = output_dir / f"{output_stem}.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        summary = {
            "input_file": str(input_file),
            "variable": args.variable,
            "time_start_requested": "" if time_start is None else time_start.isoformat(),
            "time_end_requested": "" if time_end is None else time_end.isoformat(),
            "time_start_used": time_label_start.isoformat(),
            "time_end_used": time_label_end.isoformat(),
            "height_min_requested_m": np.nan if args.all_heights else args.height_min,
            "height_max_requested_m": np.nan if args.all_heights else args.height_max,
            "height_min_used_m": float(range_values.min()),
            "height_max_used_m": float(range_values.max()),
            "all_heights": bool(args.all_heights),
            "positive_only": bool(args.positive_only),
            "density": bool(args.density),
            "bins": int(args.bins),
            "bin_width": np.nan if args.bin_width is None else float(args.bin_width),
        }
        summary.update(_summarize_values(values))

        summary_path = output_dir / f"{output_stem}_summary.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

        print(f"Input file   : {input_file}")
        print(f"Variable     : {args.variable}")
        print(
            "Time used    : "
            f"{time_label_start.strftime('%Y-%m-%d %H:%M:%S')} to "
            f"{time_label_end.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(
            "Heights used : "
            f"{range_values.min():.0f}-{range_values.max():.0f} m"
        )
        print(f"Valid values : {values.size}")
        print(f"Figure       : {figure_path}")
        print(f"Summary CSV  : {summary_path}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
