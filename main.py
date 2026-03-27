from pathlib import Path

import numpy as np
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import Range1d, RangeTool
from bokeh.plotting import figure


def load_sample(sample_dir: Path):
    time = np.load(sample_dir / "time.npy")
    ms = np.load(sample_dir / "ms.npy")
    return time, ms


def plot_sample(sample_dir: Path):
    time, ms = load_sample(sample_dir)
    time_min = time / 60.0
    tic = ms.sum(axis=1)

    mz_min = 15
    mz_max = mz_min + ms.shape[1] - 1

    # Detail view: ions heatmap, 200-scan window
    window_scans = 200
    dt = time_min[1] - time_min[0]
    window_width = window_scans * dt
    detail_range = Range1d(start=time_min[0], end=time_min[0] + window_width)

    ms_log = np.log1p(ms.T)
    p_detail = figure(title=f"Ion Heatmap — {sample_dir.name}", x_axis_label="Time (min)",
                      y_axis_label="m/z", width=1000, height=400,
                      x_range=detail_range)
    p_detail.image(image=[ms_log], x=time_min[0], y=mz_min,
                   dw=time_min[-1] - time_min[0], dh=mz_max - mz_min + 1,
                   palette="Viridis256")

    # TIC overview with range selector
    p_tic = figure(title=f"TIC — {sample_dir.name}", x_axis_label="Time (min)",
                   y_axis_label="Total Intensity", width=1000, height=250)
    p_tic.line(time_min, tic)
    range_tool = RangeTool(x_range=detail_range)
    p_tic.add_tools(range_tool)

    show(column(p_detail, p_tic))


if __name__ == "__main__":
    plot_sample(Path("data/A0"))
