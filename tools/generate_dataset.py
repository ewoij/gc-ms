"""Generate a labeled dataset of synthetic GC-MS peak clusters.

Usage:
    uv run python tools/generate_dataset.py --n 1000 --outdir data/synthetic_peaks --seed 42
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import GeneratorConfig, MoleculeConfig, NoiseConfig, generate


def load_real_distributions():
    with open("data/models/summary.json") as f:
        models = json.load(f)
    model_ids = [m["cluster_id"] for m in models]

    with open("data/spectra.json") as f:
        num_spectra = len(json.load(f))

    return model_ids, num_spectra


def sample_intensity(rng: np.random.Generator) -> float:
    # Log-uniform between ~2M and ~50M (P5–P95 from real data)
    return float(np.exp(rng.uniform(np.log(2e6), np.log(50e6))))


def sample_width(rng: np.random.Generator) -> int:
    # Half-height width P5–P95: 15–43 scans. Full profile ~2x that.
    half_width = rng.uniform(15, 43)
    return int(round(half_width * 2))


def make_sample_config(num_components: int, model_ids: list[int],
                       num_spectra: int, rng: np.random.Generator) -> GeneratorConfig:
    molecules = []
    cursor = 0  # tracks where we are in scan space

    for i in range(num_components):
        width = sample_width(rng)
        intensity = sample_intensity(rng)
        model = int(rng.choice(model_ids))
        spectrum = int(rng.integers(0, num_spectra))

        if i == 0:
            # First molecule: apex near the start, tail extends before scan 0
            apex = int(width * rng.uniform(0.1, 0.3))
        else:
            # Overlap with previous: offset by fraction of width
            overlap_factor = rng.uniform(0.3, 0.8)
            apex = cursor + int(width * overlap_factor)

        molecules.append(MoleculeConfig(
            model=model, spectrum=spectrum,
            apex=apex, width=width, intensity=intensity,
        ))
        cursor = apex

    # Compute num_scans: last molecule's apex + some tail
    last = molecules[-1]
    num_scans = last.apex + int(last.width * rng.uniform(0.3, 0.5))
    num_scans = max(num_scans, 30)  # minimum size

    return GeneratorConfig(
        num_scans=num_scans,
        noise=NoiseConfig(poisson=True, gaussian_std=100.0),
        molecules=molecules,
    )


def generate_dataset(n: int, outdir: Path, min_comp: int, max_comp: int, seed: int):
    rng = np.random.default_rng(seed)
    model_ids, num_spectra = load_real_distributions()

    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / "metadata.csv"

    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "num_components", "num_scans"])

        for i in range(n):
            sample_id = f"{i:04d}"
            num_components = int(rng.integers(min_comp, max_comp + 1))
            config = make_sample_config(num_components, model_ids, num_spectra, rng)

            sample_dir = outdir / sample_id
            generate(config, sample_dir)
            writer.writerow([sample_id, num_components, config.num_scans])

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{n}]")

    print(f"\nDataset: {n} samples -> {outdir}")
    print(f"  -> {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic GC-MS dataset")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--outdir", type=Path, default=Path("data/synthetic_peaks"))
    parser.add_argument("--min-components", type=int, default=1)
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.n, args.outdir, args.min_components, args.max_components, args.seed)


if __name__ == "__main__":
    main()
