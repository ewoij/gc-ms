"""Estimate the number of components in a GC-MS intensity matrix.

Usage:
    uv run python tools/estimate_components.py data/synthetic_peaks/0042/ms.npy
"""

import argparse
from pathlib import Path

import joblib
import numpy as np


DEFAULT_MODEL = Path("models/component_counter/model.joblib")
NUM_SVD_FEATURES = 20


def estimate_components(M: np.ndarray, model_path: Path = DEFAULT_MODEL) -> int:
    model = joblib.load(model_path)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if len(s) < NUM_SVD_FEATURES:
        s = np.pad(s, (0, NUM_SVD_FEATURES - len(s)))
    features = (s[:NUM_SVD_FEATURES] / s[0]).reshape(1, -1)
    return int(model.predict(features)[0])


def main():
    parser = argparse.ArgumentParser(description="Estimate number of components")
    parser.add_argument("ms", type=Path, help="Path to ms.npy")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    ms = np.load(args.ms)
    n = estimate_components(ms, args.model)
    print(f"Estimated components: {n}")


if __name__ == "__main__":
    main()
