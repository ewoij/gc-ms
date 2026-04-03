"""Estimate the number of components in a GC-MS intensity matrix.

Usage:
    uv run python tools/estimate_components.py data/synthetic_peaks/0042/ms.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gcms.estimator import DEFAULT_MODEL, estimate_components


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
