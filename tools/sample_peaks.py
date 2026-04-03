"""Sample peaks across runs by height, plot ions + models to test.html."""

import json
import random
import sys
from pathlib import Path

import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import Div
from bokeh.palettes import Category10, TolRainbow
from bokeh.plotting import figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gcms.pipeline import PipelineResult

# Find all results dirs
data_dir = Path("data")
results = []
for analysis_dir in sorted(data_dir.iterdir()):
    results_dir = analysis_dir / "results"
    if not results_dir.exists():
        continue
    # Pick latest results
    runs = sorted(results_dir.iterdir())
    if not runs:
        continue
    run_dir = runs[-1]
    peaks_file = run_dir / "peaks.json"
    ms_file = run_dir / "ms_clean.npy"
    if peaks_file.exists() and ms_file.exists():
        results.append((analysis_dir.name, run_dir))

print(f"Found {len(results)} analyses with results")

# Collect all peaks with metadata
all_peaks = []
for name, run_dir in results:
    ms = np.load(run_dir / "ms_clean.npy")
    tic = ms.sum(axis=1)
    result = PipelineResult.model_validate_json((run_dir / "peaks.json").read_text())
    for i, pk in enumerate(result.peaks):
        if not pk.components:
            continue
        peak_height = tic[pk.start:pk.stop].max()
        all_peaks.append({
            "name": name,
            "ms": ms,
            "peak": pk,
            "height": peak_height,
        })

print(f"Total peaks with components: {len(all_peaks)}")

# Sort by height, bin into 5 groups, sample 3 from each
all_peaks.sort(key=lambda p: p["height"])
n = len(all_peaks)
bins = [
    ("Tiny", all_peaks[:n // 5]),
    ("Small", all_peaks[n // 5:2 * n // 5]),
    ("Medium", all_peaks[2 * n // 5:3 * n // 5]),
    ("Large", all_peaks[3 * n // 5:4 * n // 5]),
    ("Huge", all_peaks[4 * n // 5:]),
]

random.seed(42)
plots = []
for bin_name, bin_peaks in bins:
    sampled = random.sample(bin_peaks, min(3, len(bin_peaks)))
    plots.append(Div(text=f"<h2>{bin_name} peaks</h2>"))

    for entry in sampled:
        pk = entry["peak"]
        ms = entry["ms"]
        name = entry["name"]
        peak_ms = ms[pk.start:pk.stop, :]
        peak_scans = np.arange(pk.start, pk.stop)
        n_comp = pk.n_components

        p = figure(title=f"{name} [{pk.start}:{pk.stop}] {n_comp}c — height={entry['height']:.0f}",
                   width=450, height=280)

        # All ions faint
        palette = TolRainbow[23]
        ion_max = peak_ms.max(axis=0)
        active_ions = [i for i in range(peak_ms.shape[1]) if ion_max[i] > 0]
        for mz in active_ions:
            p.line(peak_scans, peak_ms[:, mz], line_width=0.5, line_alpha=0.2,
                   color=palette[mz % 23])

        # Component models
        if pk.components:
            comp_palette = Category10[max(n_comp, 3)]
            models = []
            for comp in pk.components:
                profile = np.array(comp.profile)
                spectrum = np.array(comp.spectrum)
                models.append(profile * spectrum.sum())
            models = np.column_stack(models)
            if models.max() > 0:
                models = models * (peak_ms.max() / models.max())
            for i in range(models.shape[1]):
                p.line(peak_scans, models[:, i], line_width=2.5,
                       color=comp_palette[i % len(comp_palette)])

        p.title.text_font_size = "9pt"
        plots.append(p)

output_file("test.html", title="Sampled Peaks Across Runs")
save(column(*plots))
print("-> test.html")
