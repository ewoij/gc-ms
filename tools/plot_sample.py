"""Plot a GC-MS sample as interactive HTML.

Usage:
    uv run python tools/plot_sample.py data/A0
"""

import argparse
from pathlib import Path

import json

import numpy as np
from bokeh.layouts import column
from bokeh.models import Div, Range1d, RangeTool
from bokeh.plotting import figure, save, output_file


def plot_sample(sample_dir: Path):
    ms = np.load(sample_dir / "ms.npy")
    scans = np.arange(ms.shape[0])
    tic = ms.sum(axis=1)

    plots = []

    # Ground truth components (if available)
    gt_dir = sample_dir / "ground_truth"
    if gt_dir.exists():
        gt_files = sorted(gt_dir.glob("*.npy"), key=lambda p: int(p.stem))
        if gt_files:
            p_gt = figure(title=f"Input Components — {sample_dir.name}",
                          x_axis_label="Scan", y_axis_label="Intensity",
                          width=1000, height=300)
            for gt_path in gt_files:
                gt = np.load(gt_path)
                gt_tic = gt.sum(axis=1)
                p_gt.line(scans, gt_tic, legend_label=f"Molecule {gt_path.stem}")
            p_gt.legend.click_policy = "hide"
            plots.append(p_gt)

    # TIC
    p_tic = figure(title=f"TIC — {sample_dir.name}", x_axis_label="Scan",
                   y_axis_label="Intensity", width=1000, height=300,
                   x_range=plots[0].x_range if plots else None)
    p_tic.line(scans, tic)
    plots.append(p_tic)

    # Ion traces with range selector
    window_scans = min(200, ms.shape[0])
    detail_range = Range1d(start=0, end=window_scans)

    p_detail = figure(title=f"Ion Traces — {sample_dir.name}", x_axis_label="Scan",
                      y_axis_label="Intensity", width=1000, height=400,
                      x_range=detail_range)
    xs = [scans] * ms.shape[1]
    ys = [ms[:, i] for i in range(ms.shape[1])]
    p_detail.multi_line(xs, ys, line_alpha=0.3, line_width=0.5)
    plots.append(p_detail)

    # Range tool on TIC
    p_tic.add_tools(RangeTool(x_range=detail_range))

    # Config JSON (if available)
    config_path = sample_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_text = json.dumps(json.load(f), indent=2)
        plots.append(Div(text=f"<h3>Config</h3><pre>{config_text}</pre>", width=1000))

    out = sample_dir / "plot.html"
    output_file(out, title=sample_dir.name)
    save(column(*plots))
    print(f"  -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot GC-MS sample as HTML")
    parser.add_argument("dir", type=Path, help="Directory containing time.npy and ms.npy")
    args = parser.parse_args()
    plot_sample(args.dir)


if __name__ == "__main__":
    main()
