"""Generate the multi-page blog with embedded Bokeh plots."""

import csv
import json
import random
from pathlib import Path

import joblib
import numpy as np
from bokeh.embed import components
from bokeh.layouts import column, gridplot, row
from bokeh.models import Range1d, RangeTool
from bokeh.palettes import Category10, TolRainbow, Turbo256, Viridis256
from bokeh.plotting import figure
from bokeh.resources import CDN
from scipy.interpolate import interp1d
from scipy.optimize import nnls


# -- Shared HTML --

CSS = """
  body {
    max-width: 960px;
    margin: 40px auto;
    padding: 0 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background: #fafafa;
  }
  h1 { font-size: 1.8em; margin-bottom: 0.2em; }
  h2 { font-size: 1.3em; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }
  .subtitle { color: #666; margin-bottom: 2em; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
  pre { background: #f0f0f0; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }
  .plot { margin: 1.5em 0; }
  .stat { background: #e8f4f8; padding: 12px 16px; border-radius: 6px; margin: 1em 0; font-size: 0.95em; }
  .disclaimer { background: #fff3cd; padding: 12px 16px; border-radius: 6px; font-size: 0.9em; }
  a { color: #0366d6; }
  .nav { margin-bottom: 2em; font-size: 0.9em; }
  .post-list { list-style: none; padding: 0; }
  .post-list li { margin: 1.5em 0; }
  .post-list a { font-size: 1.2em; font-weight: 600; }
  .post-list p { margin: 0.3em 0 0; color: #666; }
"""

DISCLAIMER = """<p class="disclaimer">
<strong>Disclaimer:</strong> I'm learning as I go here &mdash; I have no formal background in
analytical chemistry or chemometrics. This is very much a "figure it out as you build it" project,
and nothing here should be taken as state of the art. If you spot something wrong or know a better
way, I'd love to hear about it!</p>"""


def wrap_page(title, body, nav_back=False):
    nav = ""
    if nav_back:
        nav = '<div class="nav"><a href="../index.html">&larr; All posts</a></div>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
{CDN.render()}
<style>{CSS}</style>
</head>
<body>
{nav}
{body}
</body>
</html>"""


# -- Post 1: Generator plots --

def plot_real_sample():
    ms = np.load("data/A0/ms.npy")
    scans = np.arange(ms.shape[0])
    tic = ms.sum(axis=1)

    p_tic = figure(title="TIC — A0 (Soft Camel Cheese)", x_axis_label="Scan",
                   y_axis_label="Intensity", width=900, height=250)
    p_tic.line(scans, tic)

    detail_range = Range1d(start=5100, end=5350)
    num_ions = ms.shape[1]
    colors = [TolRainbow[23][i % 23] for i in range(num_ions)]

    p_ions = figure(title="Ion Traces — a peak cluster with multiple overlapping molecules",
                    x_axis_label="Scan", y_axis_label="Intensity", width=900, height=300,
                    x_range=detail_range)
    xs = [scans] * num_ions
    ys = [ms[:, i] for i in range(num_ions)]
    p_ions.multi_line(xs, ys, line_color=colors, line_alpha=0.8, line_width=0.7)

    p_tic.add_tools(RangeTool(x_range=detail_range))
    return column(p_tic, p_ions)


def plot_peak_grid():
    peaks_dir = Path("data/A0/peak_shapes")
    all_peaks = list(peaks_dir.glob("*/*.npy"))
    big = [(p, np.load(p)) for p in all_peaks]
    big = [(p, d) for p, d in big if d.max() > 40000]
    random.seed(42)
    sampled = random.sample(big, 20)

    plots = []
    for path, peak in sampled:
        ion = path.parent.name
        p = figure(title=f"ion {ion}", width=200, height=140)
        p.line(np.arange(len(peak)), peak)
        p.title.text_font_size = "9pt"
        plots.append(p)

    return gridplot([plots[i:i+5] for i in range(0, 20, 5)], merge_tools=False)


def plot_cluster_overview():
    with open("data/clusters/index.json") as f:
        index = json.load(f)
    selected = [l.strip() for l in open("data/clusters/selection.txt") if l.strip()]

    plots = []
    for cid in selected:
        paths = index[cid]
        resampled = []
        for p in paths:
            peak = np.load(p)
            x_old = np.linspace(0, 1, len(peak))
            x_new = np.linspace(0, 1, 100)
            r = interp1d(x_old, peak)(x_new)
            mx = r.max()
            if mx > 0:
                r /= mx
            resampled.append(r)
        summed = np.sum(resampled, axis=0)
        summed /= summed.max()

        p = figure(title=f"Cluster {cid} ({len(paths)})", width=200, height=140)
        p.line(np.linspace(0, 1, 100), summed)
        p.title.text_font_size = "9pt"
        plots.append(p)

    return gridplot([plots[i:i+7] for i in range(0, len(plots), 7)], merge_tools=False)


def plot_synthetic():
    ms = np.load("data/synthetic/test4/ms.npy")
    scans = np.arange(ms.shape[0])
    tic = ms.sum(axis=1)

    with open("data/synthetic/test4/config.json") as f:
        config = json.load(f)
    with open("data/spectra.json") as f:
        spectra_lib = json.load(f)

    gt_dir = Path("data/synthetic/test4/ground_truth")
    gt_files = sorted(gt_dir.glob("*.npy"), key=lambda p: int(p.stem))
    num_mol = len(gt_files)
    mol_palette = Category10[max(num_mol, 3)]

    p_gt = figure(title="Input Components (ground truth)", x_axis_label="Scan",
                  y_axis_label="Intensity", width=900, height=250)
    for i, gt_path in enumerate(gt_files):
        gt = np.load(gt_path)
        gt_tic = gt.sum(axis=1)
        p_gt.line(scans, gt_tic, legend_label=f"Molecule {i}", color=mol_palette[i])
    p_gt.legend.click_policy = "hide"

    p_tic = figure(title="Combined TIC (with noise)", x_axis_label="Scan",
                   y_axis_label="Intensity", width=900, height=250,
                   x_range=p_gt.x_range)
    p_tic.line(scans, tic)

    num_ions = ms.shape[1]
    colors = [TolRainbow[23][i % 23] for i in range(num_ions)]
    p_ions = figure(title="Ion Traces", x_axis_label="Scan",
                    y_axis_label="Intensity", width=900, height=300,
                    x_range=p_gt.x_range)
    xs = [scans] * num_ions
    ys = [ms[:, i] for i in range(num_ions)]
    p_ions.multi_line(xs, ys, line_color=colors, line_alpha=0.8, line_width=0.7)

    spec_plots = []
    for i, mol in enumerate(config.get("molecules", [])):
        spectrum = spectra_lib[mol["spectrum"]]
        mzs = [p[0] for p in spectrum["peaks"]]
        intensities = [p[1] for p in spectrum["peaks"]]
        name = spectrum.get("name", f"Spectrum {mol['spectrum']}")
        p_spec = figure(title=f"Mol {i}: {name}", x_axis_label="m/z",
                        y_axis_label="Intensity", width=300, height=220)
        p_spec.vbar(x=mzs, top=intensities, width=0.8, color=mol_palette[i])
        p_spec.title.text_font_size = "9pt"
        spec_plots.append(p_spec)

    return column(p_gt, p_tic, p_ions, gridplot([spec_plots], merge_tools=False))


# -- Post 2: Estimator plots --

def plot_sample_components_grid():
    random.seed(42)
    samples = sorted(Path("data/synthetic_peaks").glob("*/ms.npy"))
    sampled = random.sample(samples, 12)

    plots = []
    for ms_path in sorted(sampled):
        sample_dir = ms_path.parent
        name = sample_dir.name
        ms = np.load(ms_path)
        tic = ms.sum(axis=1)
        gt_files = sorted((sample_dir / "ground_truth").glob("*.npy"), key=lambda p: int(p.stem))
        n = len(gt_files)
        palette = Category10[max(n, 3)]

        p = figure(title=f"{name} ({n}c)", width=220, height=160)
        p.line(np.arange(len(tic)), tic, color="black", line_width=1.5, line_alpha=0.4)
        for i, gt_path in enumerate(gt_files):
            gt = np.load(gt_path)
            gt_tic = gt.sum(axis=1)
            p.line(np.arange(len(gt_tic)), gt_tic, color=palette[i % len(palette)], line_alpha=0.7)
        p.title.text_font_size = "9pt"
        plots.append(p)

    return gridplot([plots[i:i+4] for i in range(0, 12, 4)], merge_tools=False)


def plot_svd_curves():
    """Show singular value decay for samples with different component counts."""
    with open("data/synthetic_peaks/metadata.csv") as f:
        rows = list(csv.DictReader(f))

    # Pick one sample per component count 1-5 and 8,10
    targets = {1: None, 2: None, 3: None, 5: None, 8: None, 10: None}
    for row in rows:
        nc = int(row["num_components"])
        if nc in targets and targets[nc] is None:
            targets[nc] = row["sample_id"]

    palette = Category10[max(len(targets), 3)]
    p = figure(title="Singular Value Decay by Component Count",
               x_axis_label="Singular Value Index", y_axis_label="Normalized Value",
               width=900, height=350, y_axis_type="log")

    for i, (nc, sid) in enumerate(sorted(targets.items())):
        if sid is None:
            continue
        ms = np.load(f"data/synthetic_peaks/{sid}/ms.npy")
        _, s, _ = np.linalg.svd(ms, full_matrices=False)
        s_norm = s[:20] / s[0]
        p.line(np.arange(20), s_norm, legend_label=f"{nc} components",
               color=palette[i % len(palette)], line_width=2)
        p.scatter(np.arange(20), s_norm, color=palette[i % len(palette)], size=4)

    p.legend.click_policy = "hide"
    p.legend.location = "top_right"
    return p


def plot_svd_grid():
    """Grid of 10 plots, one per component count, showing all SVD curves overlaid."""
    with open("data/synthetic_peaks/metadata.csv") as f:
        rows = list(csv.DictReader(f))

    # Group samples by component count
    by_count = {}
    for row in rows:
        nc = int(row["num_components"])
        by_count.setdefault(nc, []).append(row["sample_id"])

    plots = []
    for nc in range(1, 11):
        sids = by_count.get(nc, [])
        p = figure(title=f"{nc} component{'s' if nc > 1 else ''} ({len(sids)} samples)",
                   width=450, height=250, y_axis_type="log")
        if nc == 1 or nc == 6:
            p.yaxis.axis_label = "Normalized Value"
        if nc >= 6:
            p.xaxis.axis_label = "Singular Value Index"
        for sid in sids:
            ms = np.load(f"data/synthetic_peaks/{sid}/ms.npy")
            _, s, _ = np.linalg.svd(ms, full_matrices=False)
            s_norm = s[:20] / s[0]
            p.line(np.arange(20), s_norm, line_alpha=0.3, line_width=0.8, color="#1f77b4")
        plots.append(p)

    return gridplot([plots[i:i+2] for i in range(0, 10, 2)], merge_tools=False)


def plot_confusion_matrix():
    with open("models/component_counter/metrics.json") as f:
        metrics = json.load(f)

    cm = np.array(metrics["confusion_matrix"])
    labels = list(range(1, cm.shape[0] + 1))

    # Build data for rect plot
    xs, ys, vals, colors_list = [], [], [], []
    max_val = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            xs.append(labels[j])
            ys.append(labels[i])
            vals.append(str(cm[i][j]))
            intensity = cm[i][j] / max_val if max_val > 0 else 0
            ci = int(intensity * 255)
            colors_list.append(Viridis256[ci])

    p = figure(title=f"Confusion Matrix (accuracy: {metrics['accuracy']:.1%})",
               x_axis_label="Predicted", y_axis_label="Actual",
               width=500, height=500,
               x_range=[str(l) for l in labels],
               y_range=[str(l) for l in reversed(labels)])
    p.rect(x=[str(x) for x in xs], y=[str(y) for y in ys],
           width=1, height=1, color=colors_list, line_color="white")
    p.text(x=[str(x) for x in xs], y=[str(y) for y in ys],
           text=vals, text_align="center", text_baseline="middle",
           text_font_size="11pt", text_color="white")

    return p


def plot_example_peak():
    """Show a concrete example: ion traces, TIC, ground truth, and prediction."""
    sample_dir = Path("data/synthetic_peaks/0104")
    ms = np.load(sample_dir / "ms.npy")
    scans = np.arange(ms.shape[0])
    tic = ms.sum(axis=1)

    gt_files = sorted((sample_dir / "ground_truth").glob("*.npy"), key=lambda p: int(p.stem))
    num_mol = len(gt_files)
    mol_palette = Category10[max(num_mol, 3)]

    # Ion traces — this is what the instrument gives you
    num_ions = ms.shape[1]
    ion_colors = [TolRainbow[23][i % 23] for i in range(num_ions)]
    p_ions = figure(title="What the instrument sees: 301 ion channels",
                    x_axis_label="Scan", y_axis_label="Intensity",
                    width=900, height=350)
    xs = [scans] * num_ions
    ys = [ms[:, i] for i in range(num_ions)]
    p_ions.multi_line(xs, ys, line_color=ion_colors, line_alpha=0.8, line_width=0.7)

    # TIC
    p_tic = figure(title="Total Ion Chromatogram (sum of all ions)",
                   x_axis_label="Scan", y_axis_label="Intensity",
                   width=900, height=250, x_range=p_ions.x_range)
    p_tic.line(scans, tic, line_width=2)

    # Ground truth reveal
    p_gt = figure(title="The answer: 4 hidden molecules (ground truth)",
                  x_axis_label="Scan", y_axis_label="Intensity",
                  width=900, height=250, x_range=p_ions.x_range)
    p_gt.line(scans, tic, color="black", line_width=1.5, line_alpha=0.3)
    for i, gt_path in enumerate(gt_files):
        gt = np.load(gt_path)
        gt_tic = gt.sum(axis=1)
        p_gt.line(scans, gt_tic, legend_label=f"Molecule {i}", color=mol_palette[i], line_width=2)
    p_gt.legend.click_policy = "hide"

    return column(p_ions, p_tic, p_gt)


def plot_feature_importances():
    model = joblib.load("models/component_counter/model.joblib")
    imp = model.feature_importances_

    p = figure(title="Feature Importances (Random Forest)",
               x_axis_label="Singular Value Index", y_axis_label="Importance",
               width=900, height=300)
    p.vbar(x=np.arange(len(imp)), top=imp, width=0.7)
    return p


# -- Post 0: Why Deconvolute plots --

INTRO_DIR = Path("data/synthetic/blog_intro")
SPEC_IDX_0, SPEC_IDX_1 = 0, 101  # 4-METHYL-2-PENTANONE, 3-OCTANONE


def _load_intro_data():
    ms = np.load(INTRO_DIR / "ms.npy")
    gt0 = np.load(INTRO_DIR / "ground_truth/0.npy")
    gt1 = np.load(INTRO_DIR / "ground_truth/1.npy")
    with open("data/spectra.json") as f:
        spectra_lib = json.load(f)
    return ms, gt0, gt1, spectra_lib


def _make_ref_vec(spectra_lib, idx):
    vec = np.zeros(301)
    for mz, intensity in spectra_lib[idx]["peaks"]:
        if 0 <= mz <= 300:
            vec[mz] = intensity
    return vec


def _cos_sim(a, b):
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return d / n if n > 0 else 0


def plot_intro_clean_example():
    ms, gt0, gt1, _ = _load_intro_data()
    scans = np.arange(ms.shape[0])
    tic = ms.sum(axis=1)
    tic0 = gt0.sum(axis=1)
    tic1 = gt1.sum(axis=1)
    palette = Category10[3]

    # Components
    p_gt = figure(title="Two overlapping molecules (simplified — no noise)",
                  x_axis_label="Scan", y_axis_label="Intensity", width=900, height=250)
    p_gt.line(scans, tic, color="black", line_width=2, line_alpha=0.3, legend_label="Combined TIC")
    p_gt.line(scans, tic0, color=palette[0], line_width=2, legend_label="4-Methyl-2-pentanone")
    p_gt.line(scans, tic1, color=palette[1], line_width=2, legend_label="3-Octanone")
    p_gt.legend.click_policy = "hide"

    # Ion traces
    num_ions = ms.shape[1]
    ion_colors = [TolRainbow[23][i % 23] for i in range(num_ions)]
    p_ions = figure(title="All ion channels — what the instrument actually records",
                    x_axis_label="Scan", y_axis_label="Intensity", width=900, height=300,
                    x_range=p_gt.x_range)
    xs = [scans] * num_ions
    ys = [ms[:, i] for i in range(num_ions)]
    p_ions.multi_line(xs, ys, line_color=ion_colors, line_alpha=0.8, line_width=0.7)

    return column(p_gt, p_ions)


def _cosine_search(query_vec, spectra_lib, top_n=10):
    """Search spectra library by cosine similarity, return top N matches."""
    results = []
    for i, spec in enumerate(spectra_lib):
        ref = _make_ref_vec(spectra_lib, i)
        sim = _cos_sim(query_vec, ref)
        results.append((sim, i, spec.get("name", f"Spectrum {i}")))
    results.sort(key=lambda x: -x[0])
    return results[:top_n]


def _format_search_results(results):
    """Format cosine search results as HTML table."""
    rows = "".join(
        f"<tr><td>{i+1}</td><td>{name}</td><td>{sim:.4f}</td></tr>"
        for i, (sim, idx, name) in enumerate(results)
    )
    return f"""<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
<tr style="border-bottom: 2px solid #ddd;"><th>#</th><th style="text-align: left;">Molecule</th><th>Cosine Similarity</th></tr>
{rows}
</table>"""


def plot_intro_nnls_example():
    """Show NNLS on a single ion shared by both molecules."""
    ms, gt0, gt1, spectra_lib = _load_intro_data()
    profile0 = gt0.sum(axis=1)
    profile1 = gt1.sum(axis=1)
    scans = np.arange(ms.shape[0])
    palette = Category10[3]

    # Find an ion present in both molecules with good signal
    ref0 = _make_ref_vec(spectra_lib, SPEC_IDX_0)
    ref1 = _make_ref_vec(spectra_lib, SPEC_IDX_1)
    # m/z 43 is the base peak of 4-METHYL-2-PENTANONE and also present in 3-OCTANONE
    mz = 43
    ion_signal = ms[:, mz]
    ion_gt0 = gt0[:, mz]
    ion_gt1 = gt1[:, mz]

    # NNLS on this single ion
    profiles = np.column_stack([profile0, profile1])
    w, _ = nnls(profiles, ion_signal)
    fit0 = profile0 * w[0]
    fit1 = profile1 * w[1]

    p = figure(title=f"NNLS on m/z {mz}: separating one ion into two components",
               x_axis_label="Scan", y_axis_label="Intensity", width=900, height=300)
    p.line(scans, ion_signal, color="black", line_width=2, line_alpha=0.4,
           legend_label=f"Combined m/z {mz}")
    p.line(scans, fit0, color=palette[0], line_width=2,
           legend_label=f"Molecule A contribution (w={w[0]:.2f})")
    p.line(scans, fit1, color=palette[1], line_width=2,
           legend_label=f"Molecule B contribution (w={w[1]:.2f})")
    p.legend.click_policy = "hide"
    p.legend.location = "top_right"

    return p, mz, w


def plot_intro_contaminated_spectra():
    ms, gt0, gt1, spectra_lib = _load_intro_data()
    profile0 = gt0.sum(axis=1)
    profile1 = gt1.sum(axis=1)
    apex0 = int(np.argmax(profile0))
    apex1 = int(np.argmax(profile1))

    ref0 = _make_ref_vec(spectra_lib, SPEC_IDX_0)
    ref1 = _make_ref_vec(spectra_lib, SPEC_IDX_1)
    contaminated0 = ms[apex0, :]
    contaminated1 = ms[apex1, :]

    # Normalize for comparison
    def norm(v):
        mx = v.max()
        return v / mx if mx > 0 else v

    sim0 = _cos_sim(contaminated0, ref0)
    sim1 = _cos_sim(contaminated1, ref1)
    palette = Category10[3]

    # Molecule 0: reference vs contaminated
    p0_ref = figure(title=f"Reference: {spectra_lib[SPEC_IDX_0]['name']}",
                    x_axis_label="m/z", y_axis_label="Intensity",
                    width=440, height=250)
    p0_ref.vbar(x=np.arange(301), top=norm(ref0), width=0.8, color=palette[0], alpha=0.7)

    p0_cont = figure(title=f"Extracted at apex (scan {apex0}) — cos sim: {sim0:.3f}",
                     x_axis_label="m/z", y_axis_label="Intensity",
                     width=440, height=250, x_range=p0_ref.x_range)
    p0_cont.vbar(x=np.arange(301), top=norm(contaminated0), width=0.8, color="gray", alpha=0.7)

    # Molecule 1: reference vs contaminated
    p1_ref = figure(title=f"Reference: {spectra_lib[SPEC_IDX_1]['name']}",
                    x_axis_label="m/z", y_axis_label="Intensity",
                    width=440, height=250)
    p1_ref.vbar(x=np.arange(301), top=norm(ref1), width=0.8, color=palette[1], alpha=0.7)

    p1_cont = figure(title=f"Extracted at apex (scan {apex1}) — cos sim: {sim1:.3f}",
                     x_axis_label="m/z", y_axis_label="Intensity",
                     width=440, height=250, x_range=p1_ref.x_range)
    p1_cont.vbar(x=np.arange(301), top=norm(contaminated1), width=0.8, color="gray", alpha=0.7)

    return gridplot([[p0_ref, p0_cont], [p1_ref, p1_cont]], merge_tools=False)


def plot_intro_separated_spectra():
    ms, gt0, gt1, spectra_lib = _load_intro_data()
    profile0 = gt0.sum(axis=1)
    profile1 = gt1.sum(axis=1)

    ref0 = _make_ref_vec(spectra_lib, SPEC_IDX_0)
    ref1 = _make_ref_vec(spectra_lib, SPEC_IDX_1)

    # NNLS separation using ground truth profiles
    profiles = np.column_stack([profile0, profile1])
    recovered = np.zeros((2, 301))
    for mz in range(301):
        w, _ = nnls(profiles, ms[:, mz])
        recovered[0, mz] = w[0]
        recovered[1, mz] = w[1]

    def norm(v):
        mx = v.max()
        return v / mx if mx > 0 else v

    sim0 = _cos_sim(recovered[0], ref0)
    sim1 = _cos_sim(recovered[1], ref1)
    palette = Category10[3]

    p0_ref = figure(title=f"Reference: {spectra_lib[SPEC_IDX_0]['name']}",
                    x_axis_label="m/z", y_axis_label="Intensity",
                    width=440, height=250)
    p0_ref.vbar(x=np.arange(301), top=norm(ref0), width=0.8, color=palette[0], alpha=0.7)

    p0_rec = figure(title=f"Recovered after NNLS — cos sim: {sim0:.4f}",
                    x_axis_label="m/z", y_axis_label="Intensity",
                    width=440, height=250, x_range=p0_ref.x_range)
    p0_rec.vbar(x=np.arange(301), top=norm(recovered[0]), width=0.8, color=palette[0], alpha=0.7)

    p1_ref = figure(title=f"Reference: {spectra_lib[SPEC_IDX_1]['name']}",
                    x_axis_label="m/z", y_axis_label="Intensity",
                    width=440, height=250)
    p1_ref.vbar(x=np.arange(301), top=norm(ref1), width=0.8, color=palette[1], alpha=0.7)

    p1_rec = figure(title=f"Recovered after NNLS — cos sim: {sim1:.4f}",
                    x_axis_label="m/z", y_axis_label="Intensity",
                    width=440, height=250, x_range=p1_ref.x_range)
    p1_rec.vbar(x=np.arange(301), top=norm(recovered[1]), width=0.8, color=palette[1], alpha=0.7)

    return gridplot([[p0_ref, p0_rec], [p1_ref, p1_rec]], merge_tools=False)


# -- Build all pages --

def build_index():
    body = f"""
<h1>GC-MS Deconvolution Project</h1>
<p class="subtitle">Exploring chemometrics with synthetic data and machine learning</p>

{DISCLAIMER}

<ul class="post-list">
  <li>
    <a href="posts/why-deconvolute.html">Part 0: Why Deconvolute?</a>
    <p>What happens when molecules overlap, and how separating them fixes identification.</p>
  </li>
  <li>
    <a href="posts/generator.html">Part 1: Building a Synthetic GC-MS Data Generator</a>
    <p>From raw CDF files to a realistic data generator using real elution profiles and mass spectra.</p>
  </li>
  <li>
    <a href="posts/estimator.html">Part 2: Counting Components with SVD</a>
    <p>Using singular value decomposition and a random forest to estimate overlapping molecule count — 98.5% accuracy.</p>
  </li>
  <li style="opacity: 0.5;">
    <span style="font-size: 1.2em; font-weight: 600;">Part 3: Recovering Elution Profiles</span>
    <p>Upcoming</p>
  </li>
  <li style="opacity: 0.5;">
    <span style="font-size: 1.2em; font-weight: 600;">Part 4: Deconvoluting Real Peaks</span>
    <p>Upcoming &mdash; hopefully that works! 😱😂</p>
  </li>
  <li style="opacity: 0.5;">
    <span style="font-size: 1.2em; font-weight: 600;">Part 5: Benchmarking Against Existing Tools</span>
    <p>Upcoming</p>
  </li>
</ul>

<div style="background: #e8e8e8; padding: 12px 16px; border-radius: 6px; margin-top: 3em; font-size: 0.9em;">
I have to give kudos to <a href="https://claude.ai/code">Claude Code</a> here.
This whole project &mdash; parsing raw instrument files, extracting and clustering
peak shapes, building a data generator, training a model, and writing this blog &mdash;
was built in a couple of hours. The efficiency is honestly amazing and almost scary
at times.</div>

<hr style="margin-top: 3em; border: none; border-top: 1px solid #ddd;">
<p style="color: #999; font-size: 0.85em;">
  By <strong>Jonas Berdoz</strong> &middot;
  Data sources:
  <a href="https://ucphchemometrics.com/">Copenhagen Soft Camel Cheese GC-MS dataset</a>,
  <a href="https://github.com/MassBank/MassBank-data">MassBank mass spectral library</a>
</p>"""
    html = wrap_page("GC-MS Deconvolution Project", body)
    Path("index.html").write_text(html)
    print("-> index.html")


def build_why_deconvolute_post():
    p1 = plot_real_sample()
    p2 = plot_intro_clean_example()
    p3 = plot_intro_contaminated_spectra()
    p5_nnls, nnls_mz, nnls_w = plot_intro_nnls_example()
    p4 = plot_intro_separated_spectra()

    s1, d1 = components(p1)
    s2, d2 = components(p2)
    s3, d3 = components(p3)
    s5, d5 = components(p5_nnls)
    s4, d4 = components(p4)

    # Cosine search on contaminated spectra
    ms, gt0, gt1, spectra_lib = _load_intro_data()
    profile0 = gt0.sum(axis=1)
    profile1 = gt1.sum(axis=1)
    apex0 = int(np.argmax(profile0))
    apex1 = int(np.argmax(profile1))
    cont0 = ms[apex0, :]
    cont1 = ms[apex1, :]
    search_cont0 = _cosine_search(cont0, spectra_lib)
    search_cont1 = _cosine_search(cont1, spectra_lib)
    table_cont0 = _format_search_results(search_cont0)
    table_cont1 = _format_search_results(search_cont1)

    # Cosine search on recovered spectra
    profiles = np.column_stack([profile0, profile1])
    recovered = np.zeros((2, 301))
    for mz in range(301):
        w, _ = nnls(profiles, ms[:, mz])
        recovered[0, mz] = w[0]
        recovered[1, mz] = w[1]
    search_rec0 = _cosine_search(recovered[0], spectra_lib)
    search_rec1 = _cosine_search(recovered[1], spectra_lib)
    table_rec0 = _format_search_results(search_rec0)
    table_rec1 = _format_search_results(search_rec1)

    body = f"""
<h1>Why Deconvolute?</h1>
<p class="subtitle">What happens when molecules overlap, and why it matters for identification</p>

<h2>1. A Real GC-MS Run</h2>
<p>A GC-MS instrument separates molecules over time (chromatography) and measures their
mass fragmentation pattern (mass spectrometry). The result is an intensity matrix:
scans &times; m/z channels. Here's a real run from the Copenhagen Soft Camel Cheese dataset:</p>

<div class="plot">{d1}</div>
{s1}

<p>Drag the selection box on the TIC to explore different regions. The ion traces below
show hundreds of overlapping signals &mdash; this particular window contains a peak cluster
where multiple molecules are eluting at the same time. This is extremely common in GC-MS.</p>

<h2>2. A Simplified Example</h2>
<p>Let's look at what happens when two molecules coelute. Here's a clean, synthetic example
with no noise and no baseline &mdash; just two overlapping molecules:</p>

<div class="plot">{d2}</div>
{s2}

<p>The colored lines show the true elution profiles of each molecule. The instrument
doesn't see these separately &mdash; it only records the combined signal (ion channels plot).
The question is: can we identify what's in there?</p>

<h2>3. The Problem: Contaminated Spectra</h2>
<p>The standard approach to identify a molecule is to extract the mass spectrum at its
peak apex and match it against a reference library. Let's try that:</p>

<div class="plot">{d3}</div>
{s3}

<p>The left column shows the pure reference spectra from our library. The right column
shows what we actually extract from the combined signal at each apex scan. They look
similar but not identical &mdash; each extracted spectrum is <em>contaminated</em> by
the other molecule's signal.</p>

<p>What happens when we search our library of 9,971 spectra for the best match?</p>

<h3>Library search at apex of molecule A (scan {apex0}):</h3>
{table_cont0}

<h3>Library search at apex of molecule B (scan {apex1}):</h3>
{table_cont1}

<p>The correct molecule still shows up as the top match, but the scores are lower than
they should be, and the rankings could easily flip with more overlap or noisier data.
This is where things go wrong in real-world analysis.</p>

<h2>4. The Solution: Deconvolution</h2>
<p>Deconvolution is the process of separating the mixed signal back into its individual
components. The key idea:</p>
<ol>
  <li><strong>Recover the elution profiles</strong> &mdash; figure out how each molecule's
      signal varies over time</li>
  <li><strong>Separate the matrix</strong> &mdash; using the elution profiles, solve for each
      molecule's pure spectrum via NNLS (non-negative least squares)</li>
</ol>

<p>Let's see how NNLS works on a single ion. Take m/z {nnls_mz} &mdash; it's present
in both molecules. The combined signal is a mix of both elution profiles:</p>

<div class="plot">{d5}</div>
{s5}

<p>NNLS finds how much each elution profile contributes to the observed signal at this
m/z channel. In code:</p>

<pre>from scipy.optimize import nnls

# profiles: (num_scans, 2) — the two elution profiles as columns
# ion_signal: (num_scans,) — the combined signal at m/z {nnls_mz}

weights, _ = nnls(profiles, ion_signal)
# weights = [{nnls_w[0]:.2f}, {nnls_w[1]:.2f}]
# molecule A contributes {nnls_w[0]:.2f}, molecule B contributes {nnls_w[1]:.2f}</pre>

<p>The weight tells us how much of this ion belongs to each molecule. Now we simply
repeat this for <em>every</em> m/z channel (0&ndash;300). The vector of weights across
all m/z channels <em>is</em> the recovered spectrum for each molecule.</p>

<h2>5. The Payoff: Clean Spectra</h2>
<p>Using the true elution profiles (which we know in this synthetic example), NNLS
perfectly separates the mixed signal:</p>

<div class="plot">{d4}</div>
{s4}

<p>The recovered spectra match the reference <em>perfectly</em>. Let's run the library
search again on the deconvoluted spectra:</p>

<h3>Library search on recovered spectrum A:</h3>
{table_rec0}

<h3>Library search on recovered spectrum B:</h3>
{table_rec1}

<p>Perfect matches &mdash; cosine similarity of 1.0000. The molecules are now correctly
identified with no ambiguity.</p>

<h2>6. The Challenge Ahead</h2>
<p>Of course, in practice we <em>don't know</em> the elution profiles &mdash; that's the
whole problem. The upcoming posts tackle this step by step:</p>
<ul>
  <li><a href="generator.html">Part 1</a>: Building realistic synthetic training data</li>
  <li><a href="estimator.html">Part 2</a>: Estimating how many components are present (98.5% accuracy)</li>
  <li>Part 3: Recovering the elution profiles themselves</li>
  <li>Part 4: Putting it all together on real data</li>
</ul>

<hr style="margin-top: 3em; border: none; border-top: 1px solid #ddd;">
<p style="color: #999; font-size: 0.85em;">
  Data sources:
  <a href="https://ucphchemometrics.com/">Copenhagen Soft Camel Cheese GC-MS dataset</a>,
  <a href="https://github.com/MassBank/MassBank-data">MassBank mass spectral library</a>
</p>"""

    html = wrap_page("Why Deconvolute?", body, nav_back=True)
    Path("posts").mkdir(exist_ok=True)
    Path("posts/why-deconvolute.html").write_text(html)
    print("-> posts/why-deconvolute.html")


def build_generator_post():
    p1 = plot_real_sample()
    p2 = plot_peak_grid()
    p3 = plot_cluster_overview()
    p4 = plot_synthetic()

    s1, d1 = components(p1)
    s2, d2 = components(p2)
    s3, d3 = components(p3)
    s4, d4 = components(p4)

    body = f"""
<h1>Building a Synthetic GC-MS Data Generator</h1>
<p class="subtitle">From raw chromatography data to labeled training sets for component counting</p>

<h2>1. Why Synthetic Data?</h2>
<p>In GC-MS analysis, overlapping peaks are everywhere. Before you can deconvolve them,
you need to know <em>how many components</em> are hiding in each peak cluster.
Training a model to estimate this requires labeled data &mdash; and real GC-MS data
doesn't come with ground truth labels.</p>
<p>The solution: build a generator that produces realistic synthetic intensity matrices
where we control exactly how many molecules overlap and what their shapes look like.</p>

<h2>2. Starting from Real Data</h2>
<p>We started with 24 GC-MS runs from the
<a href="https://ucphchemometrics.com/">Copenhagen Soft Camel Cheese</a> dataset &mdash;
freely available ANDI-MS NetCDF (.CDF) files. Each run has 12,004 scans across m/z 15&ndash;300.</p>
<p>Extracted each CDF to numpy arrays: <code>time.npy</code> (acquisition times)
and <code>ms.npy</code> (intensity matrix, scans &times; m/z bins).</p>

<div class="stat">24 samples &middot; 12,004 scans each &middot; 286 m/z bins &middot; ~41 min acquisition time</div>

<div class="plot">{d1}</div>
{s1}

<h2>3. Extracting Peak Shapes</h2>
<p>For each of the 286 ion channels in each sample, we detected individual peaks:</p>
<ol>
  <li>Denoise with Gaussian filter (&sigma;=1.0)</li>
  <li>Detect peaks with <code>scipy.signal.find_peaks</code> (adaptive height/prominence thresholds)</li>
  <li>Find boundaries with <code>peak_widths(rel_height=0.95)</code></li>
  <li>Subtract linear baseline so each peak starts and ends at zero</li>
</ol>

<div class="stat">~200,000 peaks extracted across all 24 samples</div>

<div class="plot">{d2}</div>
{s2}

<h2>4. Clustering into Elution Profile Models</h2>
<p>We filtered to high-intensity peaks (&gt;100k) giving ~2,100 candidates,
resampled each to 100 points and normalized to unit height for shape comparison,
then clustered with k-means (30 clusters).</p>
<p>After manual review, 21 clusters were selected as clean elution profile models,
comprising 1,996 peaks. Each model's averaged profile (shown below) represents
a characteristic peak shape.</p>

<div class="stat">2,129 high-intensity peaks &rarr; 30 clusters &rarr; 21 selected models (1,996 peaks)</div>

<div class="plot">{d3}</div>
{s3}

<h2>5. Mass Spectra Library</h2>
<p>For realistic molecular fingerprints, we downloaded the
<a href="https://github.com/MassBank/MassBank-data">MassBank</a> bulk export
(NIST format, 130 MB). From 139,000 spectra, we filtered to electron impact (EI)
ionization and deduplicated by InChIKey.</p>

<div class="stat">139,006 spectra parsed &rarr; 13,473 EI spectra &rarr; 9,971 unique compounds</div>

<h2>6. The Generator</h2>
<p>Each synthetic sample is defined by a simple config:</p>
<pre>GeneratorConfig(
    num_scans=80,
    molecules=[
        MoleculeConfig(model=5,  spectrum=0,   apex=-5,  width=50, intensity=1_200_000),
        MoleculeConfig(model=2,  spectrum=50,  apex=40,  width=60, intensity=900_000),
        MoleculeConfig(model=14, spectrum=100, apex=85,  width=50, intensity=700_000),
    ]
)</pre>
<p>For each molecule, the elution profile is resampled to the desired width,
placed at the apex position, and multiplied (outer product) with the normalized
mass spectrum scaled by the target intensity. Components are summed, then
Poisson and Gaussian noise are added.</p>
<p>Apex positions can be outside the scan range to create realistic
tailing/fronting edge components.</p>

<div class="plot">{d4}</div>
{s4}

<p><a href="estimator.html">Next: Part 2 &mdash; Counting Components with SVD &rarr;</a></p>

<hr style="margin-top: 3em; border: none; border-top: 1px solid #ddd;">
<p style="color: #999; font-size: 0.85em;">
  Data sources:
  <a href="https://ucphchemometrics.com/">Copenhagen Soft Camel Cheese GC-MS dataset</a>,
  <a href="https://github.com/MassBank/MassBank-data">MassBank mass spectral library</a>
</p>"""

    html = wrap_page("Building a Synthetic GC-MS Data Generator", body, nav_back=True)
    Path("posts").mkdir(exist_ok=True)
    Path("posts/generator.html").write_text(html)
    print("-> posts/generator.html")


def build_estimator_post():
    p0 = plot_example_peak()
    p1 = plot_sample_components_grid()
    p2 = plot_svd_curves()
    p2b = plot_svd_grid()
    p3 = plot_confusion_matrix()
    p4 = plot_feature_importances()

    s0, d0 = components(p0)
    s1, d1 = components(p1)
    s2, d2 = components(p2)
    s2b, d2b = components(p2b)
    s3, d3 = components(p3)
    s4, d4 = components(p4)

    body = f"""
<h1>Counting Components with SVD</h1>
<p class="subtitle">Using singular value decomposition to estimate how many molecules overlap in a GC-MS peak</p>

<h2>1. The Problem</h2>
<p>In GC-MS, each molecule produces a unique pattern across hundreds of ion channels
as it elutes through the column. When molecules coelute (overlap in time), their signals
mix together into a tangled mess of overlapping peaks. The instrument gives you a matrix
of intensities &mdash; scans &times; m/z channels &mdash; but no indication of how many
molecules are hiding in there.</p>

<p>Here's a concrete example. This is what 4 overlapping molecules look like:</p>

<div class="plot">{d0}</div>
{s0}

<p>The top plot shows the raw ion channels &mdash; hundreds of signals overlapping in time,
each colored by m/z. The middle plot is the TIC (sum of all ions) &mdash; it looks like
one or two broad peaks, but there are actually <strong>4 molecules</strong> hiding in there
(bottom plot). Could you have guessed that from the TIC alone?</p>

<p>This is the challenge: given only the noisy intensity matrix, figure out how many
independent components are present. It's the essential first step before any
deconvolution can happen.</p>

<p>The problem scales too &mdash; here are more examples with varying numbers of components:</p>

<div class="plot">{d1}</div>
{s1}

<h2>2. SVD as Feature Extractor</h2>
<p>Singular Value Decomposition factors the intensity matrix <code>M</code> into three parts:</p>
<pre>U, s, Vt = np.linalg.svd(M, full_matrices=False)</pre>
<ul>
  <li><strong>U</strong> &mdash; elution profiles (how each component varies over time)</li>
  <li><strong>s</strong> &mdash; singular values (how &ldquo;strong&rdquo; each component is)</li>
  <li><strong>Vt</strong> &mdash; mass spectra (each component's spectral fingerprint)</li>
</ul>
<p>The key insight: for a matrix with <em>k</em> independent molecules, the first <em>k</em>
singular values will be large, and the rest will drop to noise level. The shape of this
decay curve encodes the component count.</p>

<div class="plot">{d2}</div>
{s2}

<p>The drop-off is clearly visible &mdash; a 2-component mixture has a sharp drop after s[1],
while a 10-component mixture stays elevated much longer.</p>

<p>Here's the full picture &mdash; all SVD decay curves from our dataset, grouped by
component count. Each faint line is one sample. You can see the clusters tighten as
the pattern becomes consistent within each group:</p>

<div class="plot">{d2b}</div>
{s2b}

<p>The curves clearly separate by component count, but there's overlap and noise &mdash;
especially between adjacent counts. Where exactly to draw the cutoff varies from sample
to sample. Perfect job for a classifier.</p>

<h2>3. The Model</h2>
<p>We keep it simple: extract the first 20 normalized singular values as features,
and train a <strong>random forest classifier</strong> (200 trees) to predict the
component count (1&ndash;10).</p>
<pre>features = s[:20] / s[0]  # normalized singular value decay
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)</pre>
<p>Training data: 1,000 synthetic samples generated with our
<a href="generator.html">data generator</a> (800 train / 200 test, stratified split).</p>

<h2>4. Results</h2>

<div class="stat">Test accuracy: <strong>98.5%</strong> &mdash; only 3 errors out of 200, all off-by-one</div>

<div class="plot" style="display: flex; gap: 2em; flex-wrap: wrap;">
  <div>{d3}</div>
</div>
{s3}

<p>The confusion matrix shows near-perfect diagonal &mdash; misclassifications only happen
between adjacent counts (e.g. predicting 3 instead of 2), which makes sense for
borderline cases.</p>

<h3>Feature Importances</h3>
<div class="plot">{d4}</div>
{s4}

<p>The middle singular values (s[2]&ndash;s[7]) are the most informative &mdash; s[0] is
always 1 (after normalization), s[1] is almost always high, and the late values are
mostly noise. The discriminative signal lives in the transition zone.</p>

<h2>5. Try It</h2>
<pre>from tools.estimate_components import estimate_components
import numpy as np

ms = np.load("data/synthetic_peaks/0042/ms.npy")
n = estimate_components(ms)
print(f"Estimated components: {{n}}")  # -> 4</pre>

<h2>6. What's Next</h2>
<p>Now that we can estimate <em>how many</em> components are present, the next challenge
is actually <em>separating</em> them &mdash; recovering each molecule's individual
elution profile and mass spectrum from the mixed signal.</p>

<p><a href="generator.html">&larr; Part 1: Building the Data Generator</a></p>

<hr style="margin-top: 3em; border: none; border-top: 1px solid #ddd;">
<p style="color: #999; font-size: 0.85em;">
  Built with <a href="https://scikit-learn.org">scikit-learn</a> and
  <a href="https://bokeh.org">Bokeh</a>
</p>"""

    html = wrap_page("Counting Components with SVD", body, nav_back=True)
    Path("posts").mkdir(exist_ok=True)
    Path("posts/estimator.html").write_text(html)
    print("-> posts/estimator.html")


if __name__ == "__main__":
    print("Building index...")
    build_index()
    print("Building post 0: why deconvolute...")
    build_why_deconvolute_post()
    print("Building post 1: generator...")
    build_generator_post()
    print("Building post 2: estimator...")
    build_estimator_post()
    print("Done!")
