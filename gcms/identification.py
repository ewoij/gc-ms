"""Molecule identification via cosine similarity search."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel


class Molecule(BaseModel):
    name: str
    id: str
    inchikey: Optional[str] = None
    smiles: Optional[str] = None
    instrument_type: Optional[str] = None
    formula: Optional[str] = None
    mw: Optional[int] = None
    peaks: list[tuple[int, float]]


@dataclass
class Match:
    score: float
    molecule: Molecule


def load_library(path: Path) -> list[Molecule]:
    """Load the spectra library from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [Molecule(**entry) for entry in data]


def make_ref_vec(molecule: Molecule) -> np.ndarray:
    """Convert sparse peaks to a dense 0-300 m/z vector."""
    vec = np.zeros(301)
    for mz, intensity in molecule.peaks:
        if 0 <= mz <= 300:
            vec[int(mz)] = intensity
    return vec


def _cos_sim(a, b):
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return d / n if n > 0 else 0.0


def cosine_search(query_vec, library: list[Molecule], top_n=5) -> list[Match]:
    """Search the library for the best cosine similarity matches.

    Args:
        query_vec: dense spectrum vector (0-300 m/z)
        library: list of Molecule objects
        top_n: number of top matches to return

    Returns:
        List of Match objects sorted by score descending.
    """
    results = []
    for mol in library:
        ref = make_ref_vec(mol)
        sim = _cos_sim(query_vec, ref)
        results.append(Match(score=sim, molecule=mol))
    results.sort(key=lambda m: -m.score)
    return results[:top_n]
