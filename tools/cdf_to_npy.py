"""Extract GC-MS data from ANDI-MS NetCDF (.CDF) files to numpy arrays.

Outputs:
    time.npy  — acquisition times in seconds, shape (num_scans,)
    ms.npy    — intensity matrix, shape (num_scans, mz_max - mz_min + 1)
               column i corresponds to m/z = mz_min + i
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


def extract(cdf_path: Path, outdir: Path) -> None:
    with netcdf_file(str(cdf_path), "r", mmap=False) as f:
        time = np.array(f.variables["scan_acquisition_time"].data, dtype=np.float64)

        mass_values = np.array(f.variables["mass_values"].data, dtype=np.float32)
        intensity_values = np.array(f.variables["intensity_values"].data, dtype=np.float32)
        scan_index = np.array(f.variables["scan_index"].data, dtype=np.int64)
        point_count = np.array(f.variables["point_count"].data, dtype=np.int64)

    num_scans = len(time)
    mz_rounded = np.rint(mass_values).astype(np.int32)
    mz_min, mz_max = int(mz_rounded.min()), int(mz_rounded.max())
    num_mz = mz_max - mz_min + 1

    ms = np.zeros((num_scans, num_mz), dtype=np.float32)
    for i in range(num_scans):
        start = scan_index[i]
        end = start + point_count[i]
        cols = mz_rounded[start:end] - mz_min
        ms[i, cols] = intensity_values[start:end]

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "time.npy", time)
    np.save(outdir / "ms.npy", ms)

    print(f"{cdf_path.name}: {num_scans} scans, m/z {mz_min}–{mz_max} ({num_mz} bins)")
    print(f"  -> {outdir}/time.npy  {time.shape}")
    print(f"  -> {outdir}/ms.npy    {ms.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CDF to numpy arrays")
    parser.add_argument("cdf", type=Path, help="Path to .CDF file")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory (default: same as CDF file)")
    args = parser.parse_args()

    outdir = args.outdir if args.outdir else args.cdf.parent
    extract(args.cdf, outdir)


if __name__ == "__main__":
    main()
