"""Generate synthetic GC-MS data from elution profile models and EI spectra.

Usage:
    uv run python tools/generate.py --outdir data/synthetic/run1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from scipy.interpolate import interp1d


DATA_DIR = Path("data")


class MoleculeConfig(BaseModel):
    model: int
    spectrum: int
    apex: int
    width: int
    intensity: float


class NoiseConfig(BaseModel):
    poisson: bool = True
    gaussian_std: float = 50.0


class GeneratorConfig(BaseModel):
    num_scans: int = 500
    mz_max: int = 300
    baseline: float = 1000.0
    noise: NoiseConfig = NoiseConfig()
    molecules: list[MoleculeConfig]


def load_profile(model_id: int) -> np.ndarray:
    return np.load(DATA_DIR / "models" / str(model_id) / "profile.npy")


def load_spectra() -> list[dict]:
    with open(DATA_DIR / "spectra.json") as f:
        return json.load(f)


def make_spectrum_vector(spectrum: dict, mz_max: int) -> np.ndarray:
    vec = np.zeros(mz_max + 1, dtype=np.float64)
    for mz, intensity in spectrum["peaks"]:
        if 0 <= mz <= mz_max:
            vec[mz] = intensity
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def place_profile(profile: np.ndarray, width: int, apex: int, num_scans: int) -> np.ndarray:
    # Resample profile to desired width
    x_old = np.linspace(0, 1, len(profile))
    x_new = np.linspace(0, 1, width)
    resampled = interp1d(x_old, profile)(x_new)
    resampled /= resampled.max()

    # Find the peak position in the resampled profile
    peak_idx = np.argmax(resampled)

    # Place in full-length vector centered at apex
    result = np.zeros(num_scans, dtype=np.float64)
    start = apex - peak_idx
    for i in range(width):
        scan = start + i
        if 0 <= scan < num_scans:
            result[scan] = resampled[i]

    return result


def generate(config: GeneratorConfig, outdir: Path):
    spectra = load_spectra()
    num_mz = config.mz_max + 1
    combined = np.zeros((config.num_scans, num_mz), dtype=np.float64)

    gt_dir = outdir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for i, mol in enumerate(config.molecules):
        profile = load_profile(mol.model)
        profile_vec = place_profile(profile, mol.width, mol.apex, config.num_scans)
        spectrum_vec = make_spectrum_vector(spectra[mol.spectrum], config.mz_max)

        # Outer product: (num_scans,) x (num_mz,) -> (num_scans, num_mz)
        contribution = np.outer(profile_vec * mol.intensity, spectrum_vec)

        combined += contribution
        np.save(gt_dir / f"{i}.npy", contribution.astype(np.float32))

    # Add baseline
    combined += config.baseline

    # Add noise
    if config.noise.poisson:
        combined = np.clip(combined, 0, None)
        combined = np.random.poisson(combined.astype(np.float64)).astype(np.float64)
    if config.noise.gaussian_std > 0:
        combined += np.random.normal(0, config.noise.gaussian_std, combined.shape)

    combined = np.clip(combined, 0, None)

    # Save
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "ms.npy", combined.astype(np.float32))
    time = np.linspace(0, config.num_scans * 0.2, config.num_scans)
    np.save(outdir / "time.npy", time)

    with open(outdir / "config.json", "w") as f:
        f.write(config.model_dump_json(indent=2))

    print(f"Generated: {config.num_scans} scans, {num_mz} m/z bins, {len(config.molecules)} molecules")
    print(f"  -> {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic GC-MS data")
    parser.add_argument("--outdir", type=Path, default=Path("data/synthetic/test"))
    parser.add_argument("--config", type=Path, default=None, help="Load config from JSON file")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = GeneratorConfig.model_validate_json(f.read())
    else:
        # Demo config with two overlapping molecules
        config = GeneratorConfig(
            num_scans=500,
            molecules=[
                MoleculeConfig(model=5, spectrum=0, apex=200, width=60, intensity=1_500_000),
                MoleculeConfig(model=11, spectrum=10, apex=230, width=50, intensity=800_000),
            ],
        )

    generate(config, args.outdir)


if __name__ == "__main__":
    main()
