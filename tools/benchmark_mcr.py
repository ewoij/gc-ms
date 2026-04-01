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


def benchmark_sample(sample_dir, n_components):
    ms = np.load(sample_dir / "ms.npy")
    gt_dir = sample_dir / "ground_truth"
    gt_files = sorted(gt_dir.glob("*.npy"), key=lambda p: int(p.stem))

    # True profiles and spectra
    true_profiles = []
    true_spectra = []
    for gt_path in gt_files:
        gt = np.load(gt_path)
        profile = gt.sum(axis=1)
        # Extract spectrum: use NNLS-style (profile as single column)
        # Or simpler: spectrum at the apex of this component
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

    return {
        "avg_spectra_cos": np.mean(spec_scores) if spec_scores else 0,
        "avg_profile_cos": np.mean(prof_scores) if prof_scores else 0,
        "spectra_scores": spec_scores,
        "profile_scores": prof_scores,
    }


def run_benchmark(data_dir: Path):
    with open(data_dir / "metadata.csv") as f:
        rows = list(csv.DictReader(f))

    out_dir = data_dir / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_spec_scores = []
    all_prof_scores = []
    by_count = {}
    results = []

    for i, row in enumerate(rows):
        sid = row["sample_id"]
        nc = int(row["num_components"])
        sample_dir = data_dir / sid

        try:
            result = benchmark_sample(sample_dir, nc)
        except Exception as e:
            print(f"  [{sid}] ERROR: {e}")
            continue

        results.append({
            "sample_id": sid,
            "num_components": nc,
            "avg_spectra_cos": result["avg_spectra_cos"],
            "avg_profile_cos": result["avg_profile_cos"],
        })

        all_spec_scores.extend(result["spectra_scores"])
        all_prof_scores.extend(result["profile_scores"])

        by_count.setdefault(nc, {"spec": [], "prof": []})
        by_count[nc]["spec"].extend(result["spectra_scores"])
        by_count[nc]["prof"].extend(result["profile_scores"])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(rows)}] spec={result['avg_spectra_cos']:.3f} prof={result['avg_profile_cos']:.3f}")

    # Save results CSV
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "num_components", "avg_spectra_cos", "avg_profile_cos"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    spec_arr = np.array(all_spec_scores)
    prof_arr = np.array(all_prof_scores)

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
        "by_component_count": {},
    }

    for nc in sorted(by_count):
        s = np.array(by_count[nc]["spec"])
        p = np.array(by_count[nc]["prof"])
        summary["by_component_count"][str(nc)] = {
            "n_samples": len(by_count[nc]["spec"]),
            "spectra_median": float(np.median(s)),
            "profile_median": float(np.median(p)),
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
    print(f"\nBy component count:")
    for nc in sorted(by_count):
        s = summary["by_component_count"][str(nc)]
        print(f"  {nc:>2} components: spectra={s['spectra_median']:.4f}  profiles={s['profile_median']:.4f}")
    print(f"\n  -> {out_dir}/results.csv")
    print(f"  -> {out_dir}/summary.json")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MCR-ALS")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_peaks_mcr"))
    args = parser.parse_args()
    run_benchmark(args.data)


if __name__ == "__main__":
    main()
