"""Run the GC-MS deconvolution pipeline on an analysis.

Usage:
    uv run python tools/run_pipeline.py data/A0
    uv run python tools/run_pipeline.py data/A0 --output results/my_run
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gcms.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run GC-MS deconvolution pipeline")
    parser.add_argument("dir", type=Path, help="Directory containing ms.npy")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: <dir>/results/<timestamp>)")
    parser.add_argument("--library", type=Path, default=Path("data/spectra.json"))
    args = parser.parse_args()

    # Default output dir
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.dir / "results" / timestamp
    else:
        out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    ms_raw = np.load(args.dir / "ms.npy")
    print(f"Input: {args.dir} ({ms_raw.shape[0]} scans, {ms_raw.shape[1]} m/z)")
    print(f"Output: {out_dir}")

    ms_clean, result = run_pipeline(ms_raw, args.library)

    # Save outputs
    np.save(out_dir / "ms_clean.npy", ms_clean)
    (out_dir / "peaks.json").write_text(result.model_dump_json(indent=2))

    # Summary
    total_comp = sum(p.n_components for p in result.peaks)
    print(f"\n{len(result.peaks)} peaks, {total_comp} components")
    for p in result.peaks:
        if p.components:
            top = p.components[0].matches[0]
            print(f"  [{p.start:5d}:{p.stop:5d}] {p.n_components}c → {top.molecule.name} ({top.score:.3f})")
        else:
            print(f"  [{p.start:5d}:{p.stop:5d}] 0c")

    print(f"\n-> {out_dir}/ms_clean.npy")
    print(f"-> {out_dir}/peaks.json")


if __name__ == "__main__":
    main()
