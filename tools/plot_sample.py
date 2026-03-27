"""Plot a GC-MS sample as interactive HTML.

Usage:
    uv run python tools/plot_sample.py data/A0
"""

import argparse
from pathlib import Path

import numpy as np
from bokeh.layouts import column
from bokeh.models import Range1d, RangeTool
from bokeh.plotting import figure, save, output_file


def plot_sample(sample_dir: Path):
    time = np.load(sample_dir / "time.npy")
    ms = np.load(sample_dir / "ms.npy")
    time_min = time / 60.0
    tic = ms.sum(axis=1)

    mz_min = 15

    # Detail view: individual ion traces, 200-scan window
    window_scans = 200
    dt = time_min[1] - time_min[0]
    window_width = window_scans * dt
    detail_range = Range1d(start=time_min[0], end=time_min[0] + window_width)

    p_detail = figure(title=f"Ion Traces — {sample_dir.name}", x_axis_label="Time (min)",
                      y_axis_label="Intensity", width=1000, height=400,
                      x_range=detail_range)
    xs = [time_min] * ms.shape[1]
    ys = [ms[:, i] for i in range(ms.shape[1])]
    p_detail.multi_line(xs, ys, line_alpha=0.3, line_width=0.5)

    # TIC overview with range selector
    p_tic = figure(title=f"TIC — {sample_dir.name}", x_axis_label="Time (min)",
                   y_axis_label="Total Intensity", width=1000, height=250)
    p_tic.line(time_min, tic)
    p_tic.add_tools(RangeTool(x_range=detail_range))

    out = sample_dir / "plot.html"
    output_file(out, title=sample_dir.name)
    save(column(p_detail, p_tic))
    print(f"  -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot GC-MS sample as HTML")
    parser.add_argument("dir", type=Path, help="Directory containing time.npy and ms.npy")
    args = parser.parse_args()
    plot_sample(args.dir)


if __name__ == "__main__":
    main()
