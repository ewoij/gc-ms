"""Plot a GC-MS analysis with SVD denoising to test.html."""

import numpy as np
from pathlib import Path
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

sample_dir = Path("data/A0")
ms_raw = np.load(sample_dir / "ms.npy").astype(np.float64)
scans = np.arange(ms_raw.shape[0])

ms_savgol = np.clip(savgol_filter(ms_raw, window_length=11, polyorder=3, axis=0), 0, None)
ms_gauss = np.clip(gaussian_filter1d(ms_raw, sigma=2, axis=0), 0, None)

# Pick 50 ions with strongest signal
ion_max = ms_raw.max(axis=0)
top_ions = sorted(range(ms_raw.shape[1]), key=lambda i: -ion_max[i])[:50]

plots = []
for mz in top_ions:
    p = figure(title=f"m/z {mz}", width=240, height=160)
    p.line(scans, ms_raw[:, mz], line_width=0.5, color="gray", line_alpha=0.4)
    p.line(scans, ms_savgol[:, mz], line_width=1, color="#1f77b4")
    p.line(scans, ms_gauss[:, mz], line_width=1, color="#ff7f0e")
    p.title.text_font_size = "9pt"
    plots.append(p)

grid = gridplot([plots[i:i+5] for i in range(0, len(plots), 5)], merge_tools=False)
output_file("test.html", title="GC-MS Denoising — A0 — 50 ions")
save(grid)
print("-> test.html")
