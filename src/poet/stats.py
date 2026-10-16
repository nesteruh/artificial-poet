"""Small statistics helpers: confidence intervals over seeds, smoothing."""
import numpy as np

# Two-sided 95% Student-t critical values t_{0.975, df}, df = 1..30.
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t_critical(df: int) -> float:
    return _T975[df - 1] if df <= 30 else 1.96


def mean_ci(x, axis: int = 0):
    """Mean and half-width of the 95% t-interval along `axis` (e.g. over seeds):
    mean +- t_{0.975, n-1} * s / sqrt(n)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    m = x.mean(axis=axis)
    if n < 2:
        return m, np.zeros_like(m)
    return m, t_critical(n - 1) * x.std(axis=axis, ddof=1) / np.sqrt(n)


def moving_average(x, w: int) -> np.ndarray:
    """Trailing moving average with window w (shorter window at the start)."""
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - w, 0)
    return (c[idx] - c[lo]) / (idx - lo)
