"""Build a model library from selected peak clusters.

Usage:
    uv run python tools/build_models.py data/clusters/
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


RESAMPLE_LEN = 100


def build_models(clusters_dir: Path):
    with open(clusters_dir / "index.json") as f:
        index = json.load(f)

    selected = [l.strip() for l in open(clusters_dir / "selection.txt") if l.strip()]

    models_dir = clusters_dir.parent / "models"
    summary = []

    for cid in selected:
        peak_paths = index[cid]
        model_dir = models_dir / cid
        peaks_dir = model_dir / "peaks"
        peaks_dir.mkdir(parents=True, exist_ok=True)

        resampled = []
        peak_meta = []

        for i, src in enumerate(peak_paths):
            peak = np.load(src)
            np.save(peaks_dir / f"{i}.npy", peak)

            # Parse source path: data/<sample>/peak_shapes/<ion>/<start_scan>.npy
            parts = Path(src).parts
            sample = parts[-4]
            ion = int(parts[-2])
            start_scan = int(Path(src).stem)

            peak_meta.append({
                "id": i,
                "source": src,
                "sample": sample,
                "ion": ion,
                "start_scan": start_scan,
                "length": len(peak),
                "max_intensity": float(peak.max()),
            })

            # Resample and normalize for profile
            x_old = np.linspace(0, 1, len(peak))
            x_new = np.linspace(0, 1, RESAMPLE_LEN)
            r = interp1d(x_old, peak)(x_new)
            mx = r.max()
            if mx > 0:
                r /= mx
            resampled.append(r)

        # Average profile
        profile = np.mean(resampled, axis=0).astype(np.float32)
        profile /= profile.max()
        np.save(model_dir / "profile.npy", profile)

        metadata = {
            "cluster_id": int(cid),
            "num_peaks": len(peak_paths),
            "peaks": peak_meta,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        summary.append({
            "cluster_id": int(cid),
            "num_peaks": len(peak_paths),
            "avg_length": np.mean([p["length"] for p in peak_meta]),
            "avg_max_intensity": np.mean([p["max_intensity"] for p in peak_meta]),
        })
        print(f"  Model {cid}: {len(peak_paths)} peaks")

    with open(models_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total = sum(s["num_peaks"] for s in summary)
    print(f"\n{len(selected)} models, {total} peaks total -> {models_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build model library from selected clusters")
    parser.add_argument("clusters_dir", type=Path)
    args = parser.parse_args()
    build_models(args.clusters_dir)


if __name__ == "__main__":
    main()
