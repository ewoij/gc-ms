"""Benchmark MCR-ALS on the synthetic dataset.

Usage:
    uv run python tools/benchmark_mcr.py --data data/synthetic_peaks_mcr
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from bokeh.embed import components
from bokeh.io import output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import Div
from bokeh.plotting import figure
from bokeh.resources import CDN
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gcms.mcr import mcr_als


def cos_sim(a, b):
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return d / n if n > 0 else 0


def match_components(recovered, true_list, metric_fn):
    """Match recovered components to true components using Hungarian algorithm.

    Args:
        recovered: list of recovered vectors
        true_list: list of ground truth vectors
        metric_fn: function(a, b) → similarity score

    Returns:
        list of (recovered_idx, true_idx, similarity) tuples
    """
    n = len(recovered)
    m = len(true_list)
    size = max(n, m)
    sim_matrix = np.zeros((size, size))
    for i in range(n):
        for j in range(m):
            sim_matrix[i, j] = metric_fn(recovered[i], true_list[j])

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m:
            matches.append((r, c, sim_matrix[r, c]))
    return matches


def benchmark_sample(sample_dir, n_components, result_dir=None):
    ms = np.load(sample_dir / "ms.npy")
    gt_dir = sample_dir / "ground_truth"
    gt_files = sorted(gt_dir.glob("*.npy"), key=lambda p: int(p.stem))

    # True profiles and spectra
    true_profiles = []
    true_spectra = []
    for gt_path in gt_files:
        gt = np.load(gt_path)
        profile = gt.sum(axis=1)
        apex = np.argmax(profile)
        true_spectra.append(gt[apex, :])
        true_profiles.append(profile)

    # Run MCR-ALS
    C, S = mcr_als(ms, n_components)

    # Match spectra
    recovered_spectra = [S[i, :] for i in range(S.shape[0])]
    spec_matches = match_components(recovered_spectra, true_spectra, cos_sim)

    # Match profiles
    recovered_profiles = [C[:, i] for i in range(C.shape[1])]
    prof_matches = match_components(recovered_profiles, true_profiles, cos_sim)

    spec_scores = [m[2] for m in spec_matches]
    prof_scores = [m[2] for m in prof_matches]

    # Scale ratios: use spectra matching to pair components
    scale_ratios = []
    for r_idx, t_idx, _ in spec_matches:
        rec_tic = C[:, r_idx] * S[r_idx].sum()
        true_tic = true_profiles[t_idx]
        true_sum = true_tic.sum()
        ratio = rec_tic.sum() / true_sum if true_sum > 0 else 1.0
        scale_ratios.append(ratio)

    # Save recovered components
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / "C.npy", C)
        np.save(result_dir / "S.npy", S)

    return {
        "avg_spectra_cos": np.mean(spec_scores) if spec_scores else 0,
        "avg_profile_cos": np.mean(prof_scores) if prof_scores else 0,
        "avg_scale_ratio": np.mean(scale_ratios) if scale_ratios else 1.0,
        "spectra_scores": spec_scores,
        "profile_scores": prof_scores,
        "scale_ratios": scale_ratios,
    }


def run_benchmark(data_dir: Path):
    with open(data_dir / "metadata.csv") as f:
        rows = list(csv.DictReader(f))

    out_dir = data_dir / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_spec_scores = []
    all_prof_scores = []
    all_scale_ratios = []
    by_count = {}
    results = []

    for i, row in enumerate(rows):
        sid = row["sample_id"]
        nc = int(row["num_components"])
        sample_dir = data_dir / sid

        try:
            result_dir = out_dir / sid
            result = benchmark_sample(sample_dir, nc, result_dir=result_dir)
        except Exception as e:
            print(f"  [{sid}] ERROR: {e}")
            continue

        results.append({
            "sample_id": sid,
            "num_components": nc,
            "avg_spectra_cos": result["avg_spectra_cos"],
            "avg_profile_cos": result["avg_profile_cos"],
            "avg_scale_ratio": result["avg_scale_ratio"],
        })

        all_spec_scores.extend(result["spectra_scores"])
        all_prof_scores.extend(result["profile_scores"])
        all_scale_ratios.extend(result["scale_ratios"])

        by_count.setdefault(nc, {"spec": [], "prof": [], "scale": []})
        by_count[nc]["spec"].extend(result["spectra_scores"])
        by_count[nc]["prof"].extend(result["profile_scores"])
        by_count[nc]["scale"].extend(result["scale_ratios"])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(rows)}] spec={result['avg_spectra_cos']:.3f} prof={result['avg_profile_cos']:.3f}")

    # Save results CSV
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "num_components", "avg_spectra_cos", "avg_profile_cos", "avg_scale_ratio"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    spec_arr = np.array(all_spec_scores)
    prof_arr = np.array(all_prof_scores)
    scale_arr = np.array(all_scale_ratios)

    summary = {
        "total_samples": len(results),
        "total_components": len(all_spec_scores),
        "spectra": {
            "median": float(np.median(spec_arr)),
            "p5": float(np.percentile(spec_arr, 5)),
            "p25": float(np.percentile(spec_arr, 25)),
            "p75": float(np.percentile(spec_arr, 75)),
            "p95": float(np.percentile(spec_arr, 95)),
        },
        "profiles": {
            "median": float(np.median(prof_arr)),
            "p5": float(np.percentile(prof_arr, 5)),
            "p25": float(np.percentile(prof_arr, 25)),
            "p75": float(np.percentile(prof_arr, 75)),
            "p95": float(np.percentile(prof_arr, 95)),
        },
        "scale_ratio": {
            "median": float(np.median(scale_arr)),
            "p5": float(np.percentile(scale_arr, 5)),
            "p25": float(np.percentile(scale_arr, 25)),
            "p75": float(np.percentile(scale_arr, 75)),
            "p95": float(np.percentile(scale_arr, 95)),
        },
        "by_component_count": {},
    }

    for nc in sorted(by_count):
        s = np.array(by_count[nc]["spec"])
        p = np.array(by_count[nc]["prof"])
        sc = np.array(by_count[nc]["scale"])
        summary["by_component_count"][str(nc)] = {
            "n_samples": len(by_count[nc]["spec"]),
            "spectra_median": float(np.median(s)),
            "profile_median": float(np.median(p)),
            "scale_ratio_median": float(np.median(sc)),
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"MCR-ALS Benchmark: {len(results)} samples, {len(all_spec_scores)} components")
    print(f"\nSpectra recovery (cosine similarity):")
    print(f"  Median: {summary['spectra']['median']:.4f}")
    print(f"  P5: {summary['spectra']['p5']:.4f}  P25: {summary['spectra']['p25']:.4f}  P75: {summary['spectra']['p75']:.4f}  P95: {summary['spectra']['p95']:.4f}")
    print(f"\nProfile recovery (cosine similarity):")
    print(f"  Median: {summary['profiles']['median']:.4f}")
    print(f"  P5: {summary['profiles']['p5']:.4f}  P25: {summary['profiles']['p25']:.4f}  P75: {summary['profiles']['p75']:.4f}  P95: {summary['profiles']['p95']:.4f}")
    print(f"\nScale ratio (sum(recovered) / sum(true), ideal=1.0):")
    print(f"  Median: {summary['scale_ratio']['median']:.4f}")
    print(f"  P5: {summary['scale_ratio']['p5']:.4f}  P25: {summary['scale_ratio']['p25']:.4f}  P75: {summary['scale_ratio']['p75']:.4f}  P95: {summary['scale_ratio']['p95']:.4f}")
    print(f"\nBy component count:")
    for nc in sorted(by_count):
        s = summary["by_component_count"][str(nc)]
        print(f"  {nc:>2} components: spectra={s['spectra_median']:.4f}  profiles={s['profile_median']:.4f}  scale={s['scale_ratio_median']:.4f}")
    # HTML report
    build_report(out_dir, spec_arr, prof_arr, scale_arr, by_count, results, summary)

    print(f"\n  -> {out_dir}/results.csv")
    print(f"  -> {out_dir}/summary.json")
    print(f"  -> {out_dir}/report.html")


def build_report(out_dir, spec_arr, prof_arr, scale_arr, by_count, results, summary):
    # Spectra histogram
    p_spec = figure(title="Spectra Recovery — Cosine Similarity Distribution",
                    x_axis_label="Cosine Similarity", y_axis_label="Count",
                    width=900, height=300)
    hist, edges = np.histogram(spec_arr, bins=50, range=(min(0.8, spec_arr.min()), 1.0))
    p_spec.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.7)

    # Profile histogram
    p_prof = figure(title="Profile Recovery — Cosine Similarity Distribution",
                    x_axis_label="Cosine Similarity", y_axis_label="Count",
                    width=900, height=300)
    hist, edges = np.histogram(prof_arr, bins=50, range=(min(0.8, prof_arr.min()), 1.0))
    p_prof.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.7)

    # Scale ratio histogram
    p_scale = figure(title="Scale Ratio Distribution (ideal = 1.0)",
                     x_axis_label="Scale Ratio", y_axis_label="Count",
                     width=900, height=300)
    lo = min(0.5, scale_arr.min())
    hi = max(1.5, scale_arr.max())
    hist, edges = np.histogram(scale_arr, bins=50, range=(lo, hi))
    p_scale.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.7)

    # Breakdown by component count
    counts = sorted(by_count.keys())
    spec_medians = [np.median(by_count[nc]["spec"]) for nc in counts]
    prof_medians = [np.median(by_count[nc]["prof"]) for nc in counts]

    p_by_count = figure(title="Median Cosine Similarity by Component Count",
                        x_axis_label="Number of Components", y_axis_label="Median Cosine",
                        width=900, height=300)
    p_by_count.line(counts, spec_medians, line_width=2, legend_label="Spectra", color="#1f77b4")
    p_by_count.scatter(counts, spec_medians, size=8, color="#1f77b4")
    p_by_count.line(counts, prof_medians, line_width=2, legend_label="Profiles", color="#ff7f0e")
    p_by_count.scatter(counts, prof_medians, size=8, color="#ff7f0e")
    p_by_count.legend.location = "bottom_left"

    # Scale ratio by component count
    scale_medians = [np.median(by_count[nc]["scale"]) for nc in counts]
    p_scale_by_count = figure(title="Median Scale Ratio by Component Count",
                              x_axis_label="Number of Components", y_axis_label="Median Scale Ratio",
                              width=900, height=300)
    p_scale_by_count.line(counts, scale_medians, line_width=2, color="#2ca02c")
    p_scale_by_count.scatter(counts, scale_medians, size=8, color="#2ca02c")

    # Summary stats table
    stats_html = f"""
    <h2>Summary</h2>
    <table style="border-collapse: collapse; font-size: 0.95em;">
    <tr style="border-bottom: 2px solid #ddd;">
        <th></th><th>Median</th><th>P5</th><th>P25</th><th>P75</th><th>P95</th>
    </tr>
    <tr>
        <td><strong>Spectra (cos)</strong></td>
        <td>{summary['spectra']['median']:.4f}</td>
        <td>{summary['spectra']['p5']:.4f}</td>
        <td>{summary['spectra']['p25']:.4f}</td>
        <td>{summary['spectra']['p75']:.4f}</td>
        <td>{summary['spectra']['p95']:.4f}</td>
    </tr>
    <tr>
        <td><strong>Profiles (cos)</strong></td>
        <td>{summary['profiles']['median']:.4f}</td>
        <td>{summary['profiles']['p5']:.4f}</td>
        <td>{summary['profiles']['p25']:.4f}</td>
        <td>{summary['profiles']['p75']:.4f}</td>
        <td>{summary['profiles']['p95']:.4f}</td>
    </tr>
    <tr>
        <td><strong>Scale ratio</strong></td>
        <td>{summary['scale_ratio']['median']:.4f}</td>
        <td>{summary['scale_ratio']['p5']:.4f}</td>
        <td>{summary['scale_ratio']['p25']:.4f}</td>
        <td>{summary['scale_ratio']['p75']:.4f}</td>
        <td>{summary['scale_ratio']['p95']:.4f}</td>
    </tr>
    </table>
    <p>{summary['total_samples']} samples, {summary['total_components']} components</p>
    """

    # Worst 10 samples
    sorted_results = sorted(results, key=lambda r: r["avg_spectra_cos"])
    worst_rows = "".join(
        f"<tr><td>{r['sample_id']}</td><td>{r['num_components']}</td>"
        f"<td>{r['avg_spectra_cos']:.4f}</td><td>{r['avg_profile_cos']:.4f}</td>"
        f"<td>{r['avg_scale_ratio']:.4f}</td></tr>"
        for r in sorted_results[:10]
    )
    worst_html = f"""
    <h2>Worst 10 Samples</h2>
    <table style="border-collapse: collapse; font-size: 0.95em;">
    <tr style="border-bottom: 2px solid #ddd;">
        <th>Sample</th><th>Components</th><th>Spectra Cos</th><th>Profile Cos</th><th>Scale Ratio</th>
    </tr>
    {worst_rows}
    </table>
    """

    s1, d1 = components(p_spec)
    s2, d2 = components(p_prof)
    s3, d3 = components(p_scale)
    s4, d4 = components(p_by_count)
    s5, d5 = components(p_scale_by_count)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>MCR-ALS Benchmark Report</title>
{CDN.render()}
<style>
  body {{ max-width: 960px; margin: 40px auto; padding: 0 20px;
         font-family: -apple-system, sans-serif; line-height: 1.6; color: #333; }}
  h1 {{ font-size: 1.8em; }}
  h2 {{ font-size: 1.3em; margin-top: 2em; }}
  table {{ margin: 1em 0; }}
  th, td {{ padding: 6px 16px; text-align: left; }}
</style>
</head><body>
<h1>MCR-ALS Benchmark Report</h1>
{stats_html}
{d1}{s1}
{d2}{s2}
{d3}{s3}
{d4}{s4}
{d5}{s5}
{worst_html}
</body></html>"""

    with open(out_dir / "report.html", "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MCR-ALS")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_peaks_mcr"))
    args = parser.parse_args()
    run_benchmark(args.data)


if __name__ == "__main__":
    main()
