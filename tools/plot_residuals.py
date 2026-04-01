"""Plot per-component TIC residuals for 100 random samples."""

import random
import csv
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.plotting import figure


def cos_sim(a, b):
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return d / n if n > 0 else 0


data_dir = Path("data/synthetic_peaks_mcr")
bench_dir = data_dir / "benchmark"

with open(data_dir / "metadata.csv") as f:
    rows = list(csv.DictReader(f))

random.seed(42)
sampled = random.sample(rows, 100)

plots = []
for row in sorted(sampled, key=lambda r: r["sample_id"]):
    sid = row["sample_id"]
    nc = int(row["num_components"])

    gt_files = sorted(
        (data_dir / sid / "ground_truth").glob("*.npy"),
        key=lambda p: int(p.stem),
    )
    C = np.load(bench_dir / sid / "C.npy")
    S = np.load(bench_dir / sid / "S.npy")

    # True profiles and spectra for matching
    true_profiles = []
    true_spectra = []
    for gt_path in gt_files:
        gt = np.load(gt_path)
        prof = gt.sum(axis=1)
        true_profiles.append(prof)
        true_spectra.append(gt[np.argmax(prof), :])

    # Match using Hungarian algorithm on spectra cosine
    sim = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            sim[i, j] = cos_sim(S[i], true_spectra[j])
    ri, ci = linear_sum_assignment(-sim)

    scans = np.arange(C.shape[0])
    p = figure(title=f"{sid} ({nc}c)", width=200, height=140)

    for r_idx, t_idx in zip(ri, ci):
        true_tic = true_profiles[t_idx]
        rec_tic = C[:, r_idx] * S[r_idx].sum()
        diff = true_tic - rec_tic
        p.line(scans, diff, line_alpha=0.6, line_width=0.8)

    p.title.text_font_size = "9pt"
    plots.append(p)

grid = gridplot(
    [plots[i : i + 10] for i in range(0, len(plots), 10)], merge_tools=False
)
output_file(
    "data/synthetic_peaks_mcr/benchmark/residuals_100.html",
    title="TIC Residuals (100 samples)",
)
save(grid)
print("-> data/synthetic_peaks_mcr/benchmark/residuals_100.html")
