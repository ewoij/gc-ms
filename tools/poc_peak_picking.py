"""POC: peak picking on real GC-MS data."""

import numpy as np
from pathlib import Path
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.plotting import figure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def asls_baseline(signal, lam=1e8, p=0.001, max_iter=20):
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


sample_dir = Path("data/A0")
ms_raw = np.load(sample_dir / "ms.npy").astype(np.float64)
scans = np.arange(ms_raw.shape[0])

# Denoise
ms_smooth = np.clip(gaussian_filter1d(ms_raw, sigma=2, axis=0), 0, None)

# Remove baseline per ion + clip
ms = np.zeros_like(ms_smooth)
for i in range(ms_smooth.shape[1]):
    bl = asls_baseline(ms_smooth[:, i])
    ms[:, i] = np.clip(ms_smooth[:, i] - bl, 0, None)

tic_raw = ms_raw.sum(axis=1)
tic_smooth = ms_smooth.sum(axis=1)
tic = ms.sum(axis=1)

tic_baseline = asls_baseline(tic)

p1 = figure(title="TIC — raw", x_axis_label="Scan", y_axis_label="Intensity",
            width=1000, height=300)
p1.line(scans, tic_raw, line_width=1.5, color="gray")

p2 = figure(title="TIC — denoised + baseline removed + peaks", x_axis_label="Scan",
            y_axis_label="Intensity", width=1000, height=300, x_range=p1.x_range)
p2.line(scans, tic, line_width=1.5)
p2.line(scans, tic_baseline, line_width=1.5, color="#ff7f0e")

# Peak picking
peaks, props = find_peaks(tic, prominence=2e5)
_, _, left_ips, right_ips = peak_widths(tic, peaks, rel_height=0.85)

# Filter out unreasonably wide peaks
widths = right_ips - left_ips
keep = widths < 200
peaks = peaks[keep]
props = {k: v[keep] for k, v in props.items()}
left_ips = left_ips[keep]
right_ips = right_ips[keep]

# Extend edges while signal is above baseline + threshold
above = tic - tic_baseline
threshold = tic.max() * 0.001
starts = np.zeros(len(peaks))
stops = np.zeros(len(peaks))
for i, pk in enumerate(peaks):
    j = int(left_ips[i])
    while j > 0 and above[j] > threshold and pk - j < 100:
        j -= 1
    starts[i] = j
    j = int(right_ips[i])
    while j < len(tic) - 1 and above[j] > threshold and j - pk < 100:
        j += 1
    stops[i] = j

# Resolve overlaps: split at valley between overlapping peaks
for i in range(len(peaks) - 1):
    if stops[i] > starts[i + 1]:
        valley = np.argmin(tic[peaks[i]:peaks[i + 1]]) + peaks[i]
        stops[i] = valley
        starts[i + 1] = valley

print(f"Found {len(peaks)} peaks")
for i, pk in enumerate(peaks):
    print(f"  scan={pk:5d}  start={starts[i]:7.1f}  stop={stops[i]:7.1f}  prom={props['prominences'][i]:.0f}")

# Draw boxes
from bokeh.models import BoxAnnotation
for i, pk in enumerate(peaks):
    box = BoxAnnotation(left=starts[i], right=stops[i],
                        fill_alpha=0.1, fill_color="red",
                        line_color="red", line_alpha=0.4)
    p2.add_layout(box)

p2.scatter(scans[peaks], tic[peaks], size=8, color="red", marker="inverted_triangle")

output_file("test.html", title="Peak Picking — A0")
save(column(p1, p2))
print("-> test.html")
