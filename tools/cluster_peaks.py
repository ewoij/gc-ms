"""Cluster high-intensity peak shapes by similarity.

Usage:
    uv run python tools/cluster_peaks.py data/ --min-intensity 1e5 --n-clusters 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.plotting import figure


RESAMPLE_LEN = 100


def resample_normalize(peak: np.ndarray) -> np.ndarray:
    x_old = np.linspace(0, 1, len(peak))
    x_new = np.linspace(0, 1, RESAMPLE_LEN)
    resampled = interp1d(x_old, peak)(x_new)
    mx = resampled.max()
    if mx > 0:
        resampled /= mx
    return resampled.astype(np.float32)


def collect_peaks(data_dir: Path, min_intensity: float):
    paths = sorted(data_dir.glob("*/peak_shapes/*/*.npy"))
    filtered = []
    for p in paths:
        peak = np.load(p)
        if peak.max() >= min_intensity:
            filtered.append(p)
    return filtered


def make_cluster_plot(cluster_id: int, peak_paths: list[Path], width=400, height=300):
    p = figure(title=f"Cluster {cluster_id} ({len(peak_paths)} peaks)",
               x_axis_label="Scan offset", y_axis_label="Normalized",
               width=width, height=height)
    for path in peak_paths:
        peak = np.load(path)
        normed = peak / peak.max() if peak.max() > 0 else peak
        p.line(np.linspace(0, 1, len(normed)), normed, line_alpha=0.2, line_width=0.5)
    return p


def cluster_peaks(data_dir: Path, min_intensity: float, n_clusters: int):
    print(f"Collecting peaks with intensity >= {min_intensity:.0f}...")
    paths = collect_peaks(data_dir, min_intensity)
    print(f"  {len(paths)} peaks found")

    print("Resampling for clustering...")
    features = np.array([resample_normalize(np.load(p)) for p in paths])

    print(f"Clustering into {n_clusters} groups...")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(features)

    # Build index
    clusters_dir = data_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    index = {}
    cluster_peak_paths = {}
    for cid in range(n_clusters):
        mask = labels == cid
        cid_paths = [paths[i] for i in np.where(mask)[0]]
        index[str(cid)] = [str(p) for p in cid_paths]
        cluster_peak_paths[cid] = cid_paths
        print(f"  Cluster {cid}: {len(cid_paths)} peaks")

    # Save index
    with open(clusters_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Individual cluster previews
    for cid, cid_paths in cluster_peak_paths.items():
        cid_dir = clusters_dir / str(cid)
        cid_dir.mkdir(parents=True, exist_ok=True)
        p = make_cluster_plot(cid, cid_paths)
        output_file(cid_dir / "preview.html", title=f"Cluster {cid}")
        save(p)

    # Overview grid (fresh figures to avoid document ownership conflict)
    overview_plots = [make_cluster_plot(cid, cid_paths)
                      for cid, cid_paths in cluster_peak_paths.items()]
    grid = gridplot([overview_plots[i:i+5] for i in range(0, len(overview_plots), 5)])
    out = clusters_dir / "overview.html"
    output_file(out, title="Cluster Overview")
    save(grid)
    print(f"\n  -> {clusters_dir}/index.json")
    print(f"  -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Cluster peak shapes by similarity")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--min-intensity", type=float, default=1e5)
    parser.add_argument("--n-clusters", type=int, default=30)
    args = parser.parse_args()
    cluster_peaks(args.data_dir, args.min_intensity, args.n_clusters)


if __name__ == "__main__":
    main()
