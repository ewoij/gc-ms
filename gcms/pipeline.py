"""GC-MS deconvolution pipeline."""

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from gcms.estimator import estimate_components
from gcms.identification import Molecule, cosine_search, load_library
from gcms.mcr import mcr_als
from gcms.peak_picking import pick_peaks
from gcms.preprocessing import preprocess


class ResolvedMolecule(BaseModel):
    score: float
    molecule: Molecule


class Component(BaseModel):
    spectrum: list[float]
    profile: list[float]
    matches: list[ResolvedMolecule]


class PeakResult(BaseModel):
    start: int
    stop: int
    n_components: int
    components: list[Component]


class PipelineResult(BaseModel):
    peaks: list[PeakResult]


def run_pipeline(ms_raw: np.ndarray, library_path: Path) -> tuple[np.ndarray, PipelineResult]:
    """Run the full deconvolution pipeline.

    Args:
        ms_raw: raw intensity matrix, shape (scans, mz)
        library_path: path to spectra.json

    Returns:
        (ms_clean, result) — preprocessed matrix and structured results.
    """
    ms = preprocess(ms_raw)
    peaks = pick_peaks(ms)
    library = load_library(library_path)

    peak_results = []
    for pk in peaks:
        peak_ms = ms[pk.start:pk.stop, :]
        apex_rel = pk.apex - pk.start

        n_comp = estimate_components(peak_ms)

        if n_comp == 0:
            peak_results.append(PeakResult(
                start=pk.start, stop=pk.stop, n_components=0, components=[],
            ))
            continue

        if n_comp == 1:
            peak_tic = peak_ms.sum(axis=1)
            profile = peak_tic / peak_tic.max() if peak_tic.max() > 0 else peak_tic
            C = profile.reshape(-1, 1)
            S = peak_ms[apex_rel, :].reshape(1, -1)
        else:
            C, S = mcr_als(peak_ms, n_comp)

        components = []
        for i in range(C.shape[1]):
            spec_vec = S[i, :]
            query = np.zeros(301)
            n = min(len(spec_vec), 301)
            query[:n] = spec_vec[:n]

            matches = cosine_search(query, library, top_n=5)

            components.append(Component(
                spectrum=spec_vec.tolist(),
                profile=C[:, i].tolist(),
                matches=[
                    ResolvedMolecule(score=m.score, molecule=m.molecule)
                    for m in matches
                ],
            ))

        peak_results.append(PeakResult(
            start=pk.start, stop=pk.stop, n_components=n_comp,
            components=components,
        ))

    return ms, PipelineResult(peaks=peak_results)
