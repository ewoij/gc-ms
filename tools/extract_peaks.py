"""Extract individual peak shapes from GC-MS ion chromatograms.

Usage:
    uv run python tools/extract_peaks.py data/A0

Output:
    data/A0/peak_shapes/<ion>/<start_scan>.npy — 1D float32, baseline-corrected
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


def extract_peaks(sample_dir: Path):
    ms = np.load(sample_dir / "ms.npy")
    num_scans, num_ions = ms.shape
    out_dir = sample_dir / "peak_shapes"
    total_peaks = 0

    for ion in range(num_ions):
        signal = ms[:, ion].astype(np.float64)
        if signal.max() == 0:
            continue

        smoothed = gaussian_filter1d(signal, sigma=1.0)

        nonzero = smoothed[smoothed > 0]
        if len(nonzero) == 0:
            continue
        mean_val = nonzero.mean()
        std_val = smoothed.std()

        peaks, _ = find_peaks(smoothed, height=2 * mean_val, distance=30, prominence=std_val)
        if len(peaks) == 0:
            continue

        _, _, left_ips, right_ips = peak_widths(smoothed, peaks, rel_height=0.95)

        ion_dir = out_dir / str(ion)
        ion_dir.mkdir(parents=True, exist_ok=True)

        for i, peak_idx in enumerate(peaks):
            left = int(np.floor(left_ips[i]))
            right = int(np.ceil(right_ips[i]))
            left = max(0, left)
            right = min(num_scans - 1, right)
            if right <= left:
                continue

            region = smoothed[left:right + 1].copy()
            baseline = np.linspace(region[0], region[-1], len(region))
            corrected = (region - baseline).astype(np.float32)
            np.clip(corrected, 0, None, out=corrected)

            np.save(ion_dir / f"{left}.npy", corrected)
            total_peaks += 1

    print(f"{sample_dir.name}: {total_peaks} peaks extracted -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract peak shapes from GC-MS data")
    parser.add_argument("dir", type=Path, help="Directory containing ms.npy")
    args = parser.parse_args()
    extract_peaks(args.dir)


if __name__ == "__main__":
    main()
