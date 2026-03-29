"""Parse MSP file and extract EI mass spectra into a JSON library.

Usage:
    uv run python tools/parse_msp.py tmp/MassBank_NISTformat.msp --output data/spectra.json
"""

import argparse
import json
from pathlib import Path


EI_TYPES = {"EI-B", "GC-EI-TOF", "GC-EI-FT", "GC-EI-QQ", "GC-EI-Q"}


def parse_msp(msp_path: Path):
    spectra = []
    current = {}
    peaks = []
    in_peaks = False

    with open(msp_path) as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                # End of record
                if current and peaks:
                    current["peaks"] = peaks
                    spectra.append(current)
                current = {}
                peaks = []
                in_peaks = False
                continue

            if in_peaks:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = int(round(float(parts[0])))
                        intensity = int(round(float(parts[1])))
                        peaks.append([mz, intensity])
                    except ValueError:
                        pass
                continue

            if line.startswith("Num Peaks:"):
                in_peaks = True
                continue

            if ": " in line:
                key, _, val = line.partition(": ")
                key = key.strip()
                val = val.strip()
                if key == "Name":
                    current["name"] = val
                elif key == "Formula":
                    current["formula"] = val
                elif key == "MW":
                    try:
                        current["mw"] = int(val)
                    except ValueError:
                        current["mw"] = val
                elif key == "InChIKey":
                    current["inchikey"] = val
                elif key == "SMILES":
                    current["smiles"] = val
                elif key == "Instrument_type":
                    current["instrument_type"] = val
                elif key == "DB#":
                    current["id"] = val

    # Last record
    if current and peaks:
        current["peaks"] = peaks
        spectra.append(current)

    return spectra


def main():
    parser = argparse.ArgumentParser(description="Parse MSP and extract EI spectra")
    parser.add_argument("msp", type=Path, help="Path to MSP file")
    parser.add_argument("--output", type=Path, default=Path("data/spectra.json"))
    args = parser.parse_args()

    print(f"Parsing {args.msp}...")
    all_spectra = parse_msp(args.msp)
    print(f"  {len(all_spectra)} total spectra parsed")

    ei_spectra = [s for s in all_spectra if s.get("instrument_type") in EI_TYPES]
    print(f"  {len(ei_spectra)} EI spectra")

    # Deduplicate by InChIKey (keep first)
    seen = set()
    unique = []
    for s in ei_spectra:
        key = s.get("inchikey", s.get("name", ""))
        if key and key not in seen:
            seen.add(key)
            unique.append(s)
    print(f"  {len(unique)} unique compounds (by InChIKey)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
