"""Debug: plot true vs recovered per-component TIC for a single sample."""

import csv
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.palettes import Category10_10
from bokeh.plotting import figure

SID = "0089"

data_dir = Path("data/synthetic_peaks_mcr")
bench_dir = data_dir / "benchmark"

with open(data_dir / "metadata.csv") as f:
    rows = {r["sample_id"]: r for r in csv.DictReader(f)}

row = rows[SID]
nc = int(row["num_components"])


def cos_sim(a, b):
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return d / n if n > 0 else 0


gt_files = sorted(
    (data_dir / SID / "ground_truth").glob("*.npy"),
    key=lambda p: int(p.stem),
)
C = np.load(bench_dir / SID / "C.npy")
S = np.load(bench_dir / SID / "S.npy")

# True profiles and spectra
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

p = figure(title=f"{SID} — True (solid) vs Recovered (dashed)", width=800, height=400)

for k, (r_idx, t_idx) in enumerate(zip(ri, ci)):
    true_tic = true_profiles[t_idx]
    rec_tic = C[:, r_idx] * S[r_idx].sum()
    color = Category10_10[k % 10]
    cos = sim[r_idx, t_idx]

    p.line(scans, true_tic, line_width=2, color=color,
           legend_label=f"comp {t_idx} (cos={cos:.4f})")
    p.line(scans, rec_tic, line_width=2, color=color,
           line_dash="dashed")

p.legend.click_policy = "hide"

p_diff = figure(title=f"{SID} — Residuals (true - recovered)", width=800, height=400)

for k, (r_idx, t_idx) in enumerate(zip(ri, ci)):
    true_tic = true_profiles[t_idx]
    rec_tic = C[:, r_idx] * S[r_idx].sum()
    diff = true_tic - rec_tic
    color = Category10_10[k % 10]
    p_diff.line(scans, diff, line_width=2, color=color,
                legend_label=f"comp {t_idx}")

p_diff.legend.click_policy = "hide"

output_file("data/synthetic_peaks_mcr/benchmark/debug.html",
            title=f"Debug — {SID}")
save(column(p, p_diff))
print(f"-> debug.html for {SID}")
