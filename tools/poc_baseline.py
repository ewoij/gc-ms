"""Grid of 50 strongest ions, denoised + baseline estimation."""

import numpy as np
from pathlib import Path
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from scipy.ndimage import gaussian_filter1d


from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def rolling_percentile(signal, window=101, percentile=5):
    """Compute baseline as low percentile in a moving window."""
    n = len(signal)
    half = window // 2
    baseline = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        baseline[i] = np.percentile(signal[lo:hi], percentile)
    return baseline


def asls_baseline(signal, lam=1e6, p=0.01, max_iter=20):
    """Asymmetric Least Squares baseline estimation.

    lam: smoothness (larger = smoother)
    p: asymmetry (smaller = baseline stays below signal)
    """
    n = len(signal)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    H = lam * D.T @ D
    w = np.ones(n)
    for _ in range(max_iter):
        W = diags(w, 0)
        z = spsolve(W + H, w * signal)
        w_new = np.where(signal > z, p, 1 - p)
        if np.allclose(w, w_new):
            break
        w = w_new
    return z


sample_dir = Path("data/A0")
ms_raw = np.load(sample_dir / "ms.npy").astype(np.float64)
scans = np.arange(ms_raw.shape[0])

ms = np.clip(gaussian_filter1d(ms_raw, sigma=2, axis=0), 0, None)

window = 301

# Pick 50 ions with strongest signal
ion_max = ms_raw.max(axis=0)
top_ions = sorted(range(ms_raw.shape[1]), key=lambda i: -ion_max[i])[:50]

plots = []
for mz in top_ions:
    signal = ms[:, mz]
    bl_roll = rolling_percentile(signal, window=window)
    bl_asls = asls_baseline(signal, lam=1e8, p=0.001)
    p = figure(title=f"m/z {mz}", width=240, height=160)
    p.line(scans, signal, line_width=1, color="#1f77b4")
    p.line(scans, bl_roll, line_width=1, color="#ff7f0e")
    p.line(scans, bl_asls, line_width=1, color="#2ca02c")
    p.title.text_font_size = "9pt"
    plots.append(p)

grid = gridplot([plots[i:i+5] for i in range(0, len(plots), 5)], merge_tools=False)
output_file("test.html", title="Denoised ions (Gaussian σ=2) — A0")
save(grid)
print("-> test.html")
