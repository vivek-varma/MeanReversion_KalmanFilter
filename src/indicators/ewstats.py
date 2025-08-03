import numpy as np

def ew_z(spread: np.ndarray, k: float = 0.01):
    """
    Online EW mean/var → z-score. Returns z (n,), plus the running mu, var if needed.
    """
    n   = len(spread)
    z   = np.empty(n); mu = 0.0; var = 1.0
    for i in range(n):
        x   = spread[i]
        mu  = (1 - k) * mu + k * x
        var = (1 - k) * var + k * (x - mu) ** 2
        z[i] = (x - mu) / np.sqrt(var + 1e-9)
    return z
