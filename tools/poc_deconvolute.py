"""POC: deconvolute real GC-MS peaks."""

import sys
from pathlib import Path

import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import BoxAnnotation, Div
from bokeh.palettes import Category10, TolRainbow
from bokeh.plotting import figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gcms.preprocessing import preprocess
from gcms.peak_picking import pick_peaks
from gcms.estimator import estimate_components
from gcms.mcr import mcr_als
from gcms.identification import load_library, make_ref_vec, cosine_search

# Load and preprocess
sample_dir = Path("data/A0")
ms_raw = np.load(sample_dir / "ms.npy")
print("Preprocessing...")
ms = preprocess(ms_raw)

# Peak picking
print("Peak picking...")
peaks = pick_peaks(ms)
print(f"Found {len(peaks)} peaks")

# Load spectra library
print("Loading spectra library...")
library = load_library(Path("data/spectra.json"))
print(f"Library: {len(library)} molecules")

# TIC overview
scans = np.arange(ms.shape[0])
tic = ms.sum(axis=1)

p_tic = figure(title="TIC — A0", x_axis_label="Scan", y_axis_label="Intensity",
               width=1000, height=300)
p_tic.line(scans, tic, line_width=1.5)

# Pick top 5 peaks by prominence
top_peaks = sorted(peaks, key=lambda p: -p.prominence)[:50]
top_peaks.sort(key=lambda p: p.apex)

for pk in top_peaks:
    box = BoxAnnotation(left=pk.start, right=pk.stop,
                        fill_alpha=0.1, fill_color="red",
                        line_color="red", line_alpha=0.4)
    p_tic.add_layout(box)
p_tic.scatter([pk.apex for pk in top_peaks], [tic[pk.apex] for pk in top_peaks],
              size=8, color="red", marker="inverted_triangle")

plots = [p_tic]

# Deconvolve each peak
for pk in top_peaks:
    peak_ms = ms[pk.start:pk.stop, :]
    apex_rel = pk.apex - pk.start
    peak_scans = np.arange(pk.start, pk.stop)

    n_comp = estimate_components(peak_ms)
    print(f"\nPeak scan={pk.apex} [{pk.start}:{pk.stop}] → {n_comp} component(s)")

    plots.append(Div(text=f"<h2>Peak at scan {pk.apex} ({pk.stop - pk.start} scans, {n_comp} component{'s' if n_comp != 1 else ''})</h2>"))

    # Ion traces
    if n_comp == 0:
        continue

    if n_comp == 1:
        peak_tic = peak_ms.sum(axis=1)
        # Normalize profile to peak=1 so C * S[mz] matches ion scale
        profile = peak_tic / peak_tic.max()
        C = profile.reshape(-1, 1)
        S = peak_ms[apex_rel, :].reshape(1, -1)
        spectra = [S[0, :]]
    else:
        C, S = mcr_als(peak_ms, n_comp)
        spectra = [S[i, :] for i in range(S.shape[0])]

    comp_palette = Category10[max(n_comp, 3)]

    # Combined plot: all ions (faint) + recovered profiles on their best ion
    p_combined = figure(title=f"Ion traces + recovered profiles",
                        x_axis_label="Scan", y_axis_label="Intensity",
                        width=1000, height=400)

    # All active ions as faint background
    ion_max = peak_ms.max(axis=0)
    active_ions = [i for i in range(peak_ms.shape[1]) if ion_max[i] > 0]
    palette = TolRainbow[23]
    for mz in active_ions:
        p_combined.line(peak_scans, peak_ms[:, mz], line_width=0.5, line_alpha=0.2,
                        color=palette[mz % 23])

    # Overlay per-component TIC models, scaled to match the ion range
    if C is not None:
        # Build per-component TIC: sum across m/z
        models = np.column_stack([
            (C[:, [i]] @ S[[i], :]).sum(axis=1) for i in range(C.shape[1])
        ])
        # Scale so models max matches data max (preserves ratios between components)
        scale = peak_ms.max() / models.max() if models.max() > 0 else 1.0
        models = models * scale

        for i in range(models.shape[1]):
            color = comp_palette[i % len(comp_palette)]
            p_combined.line(peak_scans, models[:, i], line_width=2.5, color=color,
                            legend_label=f"Comp {i}")
    p_combined.legend.click_policy = "hide"
    plots.append(p_combined)

    # Per-component: identification table
    match_rows = ""
    for i, spec_vec in enumerate(spectra):
        query = np.zeros(301)
        n = min(len(spec_vec), 301)
        query[:n] = spec_vec[:n]

        matches = cosine_search(query, library, top_n=5)
        color = comp_palette[i % len(comp_palette)] if n_comp > 1 else "#1f77b4"

        match_rows += f'<tr><td colspan="3" style="border-top:2px solid #ddd; padding-top:8px;"><strong style="color:{color}">Component {i}</strong></td></tr>'
        for m in matches:
            match_rows += f'<tr><td>{m.score:.3f}</td><td>{m.molecule.name}</td><td>{m.molecule.formula or ""}</td></tr>'

        for m in matches:
            print(f"    comp {i}: {m.score:.4f}  {m.molecule.name} ({m.molecule.formula})")

    table_html = f'<table style="font-size:0.9em; border-collapse:collapse;"><tr style="border-bottom:1px solid #999;"><th>Cosine</th><th>Molecule</th><th>Formula</th></tr>{match_rows}</table>'
    plots.append(Div(text=table_html, width=1000))

output_file("test.html", title="Deconvolution — A0")
save(column(*plots))
print("\n-> test.html")
