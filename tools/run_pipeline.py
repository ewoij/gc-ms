"""Run the GC-MS deconvolution pipeline on an analysis.

Usage:
    uv run python tools/run_pipeline.py data/A0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gcms.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run GC-MS deconvolution pipeline")
    parser.add_argument("dir", type=Path, help="Directory containing ms.npy")
    parser.add_argument("--library", type=Path, default=Path("data/spectra.json"))
    args = parser.parse_args()

    ms_raw = np.load(args.dir / "ms.npy")
    print(f"Input: {args.dir} ({ms_raw.shape[0]} scans, {ms_raw.shape[1]} m/z)")

    ms_clean, result = run_pipeline(ms_raw, args.library)

    # Save outputs
    np.save(args.dir / "ms_clean.npy", ms_clean)

    out_path = args.dir / "peaks.json"
    out_path.write_text(result.model_dump_json(indent=2))

    # Summary
    total_comp = sum(p.n_components for p in result.peaks)
    print(f"\n{len(result.peaks)} peaks, {total_comp} components")
    for p in result.peaks:
        if p.components:
            top = p.components[0].matches[0]
            print(f"  [{p.start:5d}:{p.stop:5d}] {p.n_components}c → {top.molecule.name} ({top.score:.3f})")
        else:
            print(f"  [{p.start:5d}:{p.stop:5d}] 0c")

    print(f"\n-> {args.dir}/ms_clean.npy")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
