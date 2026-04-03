"""GC-MS data preprocessing: denoising and baseline removal."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def asls_baseline(signal, lam=1e8, p=0.001, max_iter=20):
    """Asymmetric Least Squares baseline estimation.

    Args:
        signal: 1D signal array
        lam: smoothness (larger = smoother)
        p: asymmetry (smaller = baseline stays below signal)
        max_iter: maximum iterations

    Returns:
        Estimated baseline array.
    """
    n = len(signal)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), dtype=float)
    H = lam * D.T @ D
    w = np.ones(n)
    for _ in range(max_iter):
        W = diags(w, 0, format="csc")
        z = spsolve(W + H, w * signal)
        w_new = np.where(signal > z, p, 1 - p)
        if np.allclose(w, w_new):
            break
        w = w_new
    return z


def preprocess(ms_raw):
    """Denoise and remove baseline from a raw intensity matrix.

    Args:
        ms_raw: raw intensity matrix, shape (scans, mz)

    Returns:
        Cleaned intensity matrix, same shape.
    """
    ms_raw = ms_raw.astype(np.float64)

    # Denoise with Gaussian filter
    ms_smooth = np.clip(gaussian_filter1d(ms_raw, sigma=2, axis=0), 0, None)

    # Remove baseline per ion channel
    ms = np.zeros_like(ms_smooth)
    for i in range(ms_smooth.shape[1]):
        bl = asls_baseline(ms_smooth[:, i])
        ms[:, i] = np.clip(ms_smooth[:, i] - bl, 0, None)

    return ms
