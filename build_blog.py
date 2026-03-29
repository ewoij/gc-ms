"""Generate the blog post HTML with embedded Bokeh plots."""

import json
import random
from pathlib import Path

import numpy as np
from bokeh.embed import components
from bokeh.layouts import column, gridplot
from bokeh.models import Range1d, RangeTool
from bokeh.palettes import Category10, Turbo256
from bokeh.plotting import figure
from bokeh.resources import CDN
from scipy.interpolate import interp1d


def plot_real_sample():
    ms = np.load("data/A0/ms.npy")
    scans = np.arange(ms.shape[0])
    tic = ms.sum(axis=1)

    p_tic = figure(title="TIC — A0 (Soft Camel Cheese)", x_axis_label="Scan",
                   y_axis_label="Intensity", width=900, height=250)
    p_tic.line(scans, tic)

    detail_range = Range1d(start=0, end=200)
    num_ions = ms.shape[1]
    colors = [Turbo256[int(i * 255 / max(num_ions - 1, 1))] for i in range(num_ions)]

    p_ions = figure(title="Ion Traces (200-scan window)", x_axis_label="Scan",
                    y_axis_label="Intensity", width=900, height=300,
                    x_range=detail_range)
    xs = [scans] * num_ions
    ys = [ms[:, i] for i in range(num_ions)]
    p_ions.multi_line(xs, ys, line_color=colors, line_alpha=0.4, line_width=0.5)

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

    # Components
    p_gt = figure(title="Input Components (ground truth)", x_axis_label="Scan",
                  y_axis_label="Intensity", width=900, height=250)
    for i, gt_path in enumerate(gt_files):
        gt = np.load(gt_path)
        gt_tic = gt.sum(axis=1)
        p_gt.line(scans, gt_tic, legend_label=f"Molecule {i}", color=mol_palette[i])
    p_gt.legend.click_policy = "hide"

    # TIC
    p_tic = figure(title="Combined TIC (with noise)", x_axis_label="Scan",
                   y_axis_label="Intensity", width=900, height=250,
                   x_range=p_gt.x_range)
    p_tic.line(scans, tic)

    # Ion traces
    num_ions = ms.shape[1]
    colors = [Turbo256[int(i * 255 / max(num_ions - 1, 1))] for i in range(num_ions)]
    p_ions = figure(title="Ion Traces", x_axis_label="Scan",
                    y_axis_label="Intensity", width=900, height=300,
                    x_range=p_gt.x_range)
    xs = [scans] * num_ions
    ys = [ms[:, i] for i in range(num_ions)]
    p_ions.multi_line(xs, ys, line_color=colors, line_alpha=0.4, line_width=0.5)

    # Spectra
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


def build_html():
    p1 = plot_real_sample()
    p2 = plot_peak_grid()
    p3 = plot_cluster_overview()
    p4 = plot_synthetic()

    s1, d1 = components(p1)
    s2, d2 = components(p2)
    s3, d3 = components(p3)
    s4, d4 = components(p4)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Building a Synthetic GC-MS Data Generator</title>
{CDN.render()}
<style>
  body {{
    max-width: 960px;
    margin: 40px auto;
    padding: 0 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background: #fafafa;
  }}
  h1 {{ font-size: 1.8em; margin-bottom: 0.2em; }}
  h2 {{ font-size: 1.3em; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }}
  .subtitle {{ color: #666; margin-bottom: 2em; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: #f0f0f0; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }}
  .plot {{ margin: 1.5em 0; }}
  .stat {{ background: #e8f4f8; padding: 12px 16px; border-radius: 6px; margin: 1em 0; font-size: 0.95em; }}
  a {{ color: #0366d6; }}
</style>
</head>
<body>

<h1>Building a Synthetic GC-MS Data Generator</h1>
<p class="subtitle">From raw chromatography data to labeled training sets for component counting</p>

<p style="background: #fff3cd; padding: 12px 16px; border-radius: 6px; font-size: 0.9em;">
<strong>Disclaimer:</strong> I'm learning as I go here &mdash; I have no formal background in
analytical chemistry or chemometrics. This is very much a "figure it out as you build it" project,
and nothing here should be taken as state of the art. If you spot something wrong or know a better
way, I'd love to hear about it!</p>

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

<h2>7. What's Next</h2>
<p>With the generator in place, the next steps are:</p>
<ul>
  <li>Generate a large training dataset with varying numbers of components (1&ndash;10+),
      different degrees of overlap, noise levels, and intensities</li>
  <li>Train a model to predict the number of components from the intensity matrix</li>
  <li>Use component count estimation as the first step in a deconvolution pipeline</li>
</ul>

<hr style="margin-top: 3em; border: none; border-top: 1px solid #ddd;">
<p style="color: #999; font-size: 0.85em;">
  Data sources:
  <a href="https://ucphchemometrics.com/">Copenhagen Soft Camel Cheese GC-MS dataset</a>,
  <a href="https://github.com/MassBank/MassBank-data">MassBank mass spectral library</a>
</p>

</body>
</html>"""

    with open("index.html", "w") as f:
        f.write(html)
    print("-> index.html")


if __name__ == "__main__":
    build_html()
