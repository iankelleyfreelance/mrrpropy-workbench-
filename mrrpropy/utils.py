from datetime import datetime
import numpy as np

def to_time_slice(target: datetime | tuple[datetime, datetime] | slice) -> slice:
    if isinstance(target, slice):
        return target
    if isinstance(target, tuple):
        t0, t1 = target
        if t0 >= t1:
            raise ValueError("target_datetime tuple must be increasing (start, end).")
        return slice(np.datetime64(t0), np.datetime64(t1))
    # single datetime -> a tiny slice around it (nearest semantics later)
    t = np.datetime64(target)
    return slice(t, t)


def ols_slope_intercept_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    OLS for y = a + b x.
    Returns (b, a, r2). Requires len(x) >= 2 and finite values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()

    dx = x - x_mean
    dy = y - y_mean

    sxx = np.dot(dx, dx)
    if sxx == 0.0:
        return np.nan, np.nan, np.nan

    b = np.dot(dx, dy) / sxx
    a = y_mean - b * x_mean

    # R^2
    y_hat = a + b * x
    ss_res = np.dot(y - y_hat, y - y_hat)
    ss_tot = np.dot(y - y_mean, y - y_mean)
    if ss_tot == 0.0:
        # y constant -> define r2 as 1 if perfect fit, else nan; here residual also 0 => 1
        return b, a, 1.0 if ss_res == 0.0 else np.nan

    r2 = 1.0 - (ss_res / ss_tot)
    return b, a, r2


def compute_eps(values: np.ndarray, q: float) -> float:
    """Quantile-based epsilon over positive finite values."""
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        return np.nan
    return float(np.quantile(v, q))


def sign_from_center(c: np.ndarray, tol: float = 0.05) -> np.ndarray:
    # c en [0,1], centro 0.5
    s = np.zeros_like(c, dtype=int)
    s[c > 0.5 + tol] = +1
    s[c < 0.5 - tol] = -1
    return s

def strength(c: np.ndarray) -> np.ndarray:
    # 0..1
    return np.clip(np.abs(c - 0.5) / 0.5, 0.0, 1.0)
