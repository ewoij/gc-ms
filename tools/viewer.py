"""Interactive viewer for pipeline results.

Usage:
    panel serve tools/viewer.py --show --args data/A0/results/20260403_150707
"""

import sys
from pathlib import Path

import numpy as np
import panel as pn
from bokeh.events import Tap
from bokeh.models import BoxAnnotation, ColumnDataSource, Span, TapTool
from bokeh.palettes import Category10, TolRainbow
from bokeh.plotting import figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gcms.pipeline import PipelineResult

pn.extension()

# Load data
results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
ms = np.load(results_dir / "ms_clean.npy")
result = PipelineResult.model_validate_json((results_dir / "peaks.json").read_text())
peaks = result.peaks

scans = np.arange(ms.shape[0])
tic = ms.sum(axis=1)

# State
peak_idx = pn.widgets.IntInput(value=0, start=0, end=len(peaks) - 1, visible=False)

# Navigation
btn_prev = pn.widgets.Button(name="← Previous", button_type="default", width=100)
btn_next = pn.widgets.Button(name="Next →", button_type="default", width=100)
peak_label = pn.pane.Markdown("", width=600)


def on_prev(event):
    if peak_idx.value > 0:
        peak_idx.value -= 1


def on_next(event):
    if peak_idx.value < len(peaks) - 1:
        peak_idx.value += 1


btn_prev.on_click(on_prev)
btn_next.on_click(on_next)

# TIC plot — static, with clickable peak markers
p_tic = figure(title="TIC", x_axis_label="Scan", y_axis_label="Intensity",
               width=1000, height=250, tools="pan,wheel_zoom,box_zoom,reset")
p_tic.line(scans, tic, line_width=1)

# Peak boxes (stored for highlight updates)
peak_boxes = []
for i, pk in enumerate(peaks):
    box = BoxAnnotation(left=pk.start, right=pk.stop,
                        fill_alpha=0.08, fill_color="red",
                        line_color="red", line_alpha=0.4)
    p_tic.add_layout(box)
    peak_boxes.append(box)

# Selected peak highlight
sel_box = BoxAnnotation(fill_alpha=0.2, fill_color="#3b82f6",
                        line_color="#3b82f6", line_alpha=0.6)
p_tic.add_layout(sel_box)

# Clickable scatter markers
peak_source = ColumnDataSource(data={
    "x": [pk.start + (pk.stop - pk.start) / 2 for pk in peaks],
    "y": [tic[pk.start:pk.stop].max() for pk in peaks],
    "idx": list(range(len(peaks))),
})
p_tic.scatter("x", "y", source=peak_source, size=10, color="red",
              marker="inverted_triangle", alpha=0.7)
p_tic.add_tools(TapTool())

# Handle click on TIC — find nearest peak
def on_tic_tap(event):
    x = event.x
    # Find closest peak by center
    centers = [(pk.start + pk.stop) / 2 for pk in peaks]
    closest = min(range(len(centers)), key=lambda i: abs(centers[i] - x))
    peak_idx.value = closest

p_tic.on_event(Tap, on_tic_tap)

# Update TIC highlight
def update_tic_highlight(idx):
    pk = peaks[idx]
    sel_box.left = pk.start
    sel_box.right = pk.stop

tic_pane = pn.pane.Bokeh(p_tic)


# Detail: ion traces + component models
def make_detail_plot(idx):
    pk = peaks[idx]
    peak_ms = ms[pk.start:pk.stop, :]
    peak_scans = np.arange(pk.start, pk.stop)
    n_comp = pk.n_components

    p = figure(title=f"Peak [{pk.start}:{pk.stop}] — {n_comp} component(s)",
               x_axis_label="Scan", y_axis_label="Intensity",
               width=1000, height=350, tools="pan,wheel_zoom,box_zoom,reset")

    palette = TolRainbow[23]
    ion_max = peak_ms.max(axis=0)
    active_ions = [i for i in range(peak_ms.shape[1]) if ion_max[i] > 0]
    for mz in active_ions:
        p.line(peak_scans, peak_ms[:, mz], line_width=0.5, line_alpha=0.2,
               color=palette[mz % 23])

    if pk.components:
        comp_palette = Category10[max(n_comp, 3)]
        models = []
        for comp in pk.components:
            profile = np.array(comp.profile)
            spectrum = np.array(comp.spectrum)
            model_tic = profile * spectrum.sum()
            models.append(model_tic)

        models = np.column_stack(models) if models else np.zeros((len(peak_scans), 1))
        if models.max() > 0:
            scale = peak_ms.max() / models.max()
            models = models * scale

        for i in range(models.shape[1]):
            color = comp_palette[i % len(comp_palette)]
            p.line(peak_scans, models[:, i], line_width=2.5, color=color,
                   legend_label=f"Comp {i}")
        p.legend.click_policy = "hide"

    return p


# Match table
def make_match_table(idx):
    pk = peaks[idx]
    if not pk.components:
        return "<p><em>No components</em></p>"

    n_comp = pk.n_components
    comp_palette = Category10[max(n_comp, 3)]

    rows = ""
    for i, comp in enumerate(pk.components):
        color = comp_palette[i % len(comp_palette)] if n_comp > 1 else "#1f77b4"
        rows += f'<tr><td colspan="3" style="border-top:2px solid #ddd; padding-top:8px;"><strong style="color:{color}">Component {i}</strong></td></tr>'
        for m in comp.matches:
            rows += f'<tr><td>{m.score:.3f}</td><td>{m.molecule.name}</td><td>{m.molecule.formula or ""}</td></tr>'

    return f"""<table style="font-size:0.9em; border-collapse:collapse; width:100%;">
    <tr style="border-bottom:1px solid #999;"><th>Cosine</th><th>Molecule</th><th>Formula</th></tr>
    {rows}</table>"""


# Reactive detail panel
detail_plot = pn.pane.Bokeh(make_detail_plot(0))
match_table = pn.pane.HTML(make_match_table(0), width=1000)


def on_peak_change(event):
    idx = event.new
    pk = peaks[idx]
    peak_label.object = f"**Peak {idx + 1}/{len(peaks)}** — scan {pk.start}:{pk.stop} — {pk.n_components} component(s)"
    update_tic_highlight(idx)
    detail_plot.object = make_detail_plot(idx)
    match_table.object = make_match_table(idx)


peak_idx.param.watch(on_peak_change, "value")

# Initialize
update_tic_highlight(0)
pk0 = peaks[0]
peak_label.object = f"**Peak 1/{len(peaks)}** — scan {pk0.start}:{pk0.stop} — {pk0.n_components} component(s)"

# Layout
nav_bar = pn.Row(btn_prev, peak_label, btn_next, align="center")
layout = pn.Column(tic_pane, nav_bar, detail_plot, match_table)
layout.servable(title="GC-MS Viewer")
