"""Plot a random sample of extracted peak shapes.

Usage:
    uv run python tools/plot_peaks.py data/A0 --n 500
"""

import argparse
import random
from pathlib import Path

import numpy as np
from bokeh.io import output_file, save
from bokeh.plotting import figure


def plot_peaks(sample_dir: Path, n: int):
    peak_dir = sample_dir / "peak_shapes"
    all_peaks = list(peak_dir.glob("*/*.npy"))
    if len(all_peaks) == 0:
        print("No peaks found")
        return

    sampled = random.sample(all_peaks, min(n, len(all_peaks)))

    p = figure(title=f"Peak Shapes — {sample_dir.name} ({len(sampled)} sampled)",
               x_axis_label="Scan offset", y_axis_label="Intensity",
               width=1000, height=500)

    for path in sampled:
        peak = np.load(path)
        # Normalize to [0, 1] for comparison
        peak_max = peak.max()
        if peak_max > 0:
            normed = peak / peak_max
        else:
            continue
        p.line(np.arange(len(normed)), normed, line_alpha=0.15, line_width=0.5)

    out = sample_dir / "peaks_sample.html"
    output_file(out, title=f"Peak Shapes — {sample_dir.name}")
    save(p)
    print(f"  -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot sampled peak shapes")
    parser.add_argument("dir", type=Path)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    plot_peaks(args.dir, args.n)


if __name__ == "__main__":
    main()
