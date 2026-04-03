"""Peak picking on GC-MS TIC."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths

from gcms.preprocessing import asls_baseline


@dataclass
class Peak:
    apex: int
    start: int
    stop: int
    prominence: float


def pick_peaks(ms, prominence=2e5, max_width=200, edge_threshold=0.001,
               max_half_width=100):
    """Pick peaks from a preprocessed intensity matrix.

    Args:
        ms: preprocessed intensity matrix, shape (scans, mz)
        prominence: minimum peak prominence for detection
        max_width: discard peaks wider than this (scans)
        edge_threshold: extend edges while tic - baseline > tic.max() * this
        max_half_width: max extension from apex in each direction

    Returns:
        List of Peak objects, sorted by scan position.
    """
    tic = ms.sum(axis=1)
    baseline = asls_baseline(tic)

    peaks_idx, props = find_peaks(tic, prominence=prominence)
    _, _, left_ips, right_ips = peak_widths(tic, peaks_idx, rel_height=0.85)

    # Filter out unreasonably wide peaks
    widths = right_ips - left_ips
    keep = widths < max_width
    peaks_idx = peaks_idx[keep]
    props = {k: v[keep] for k, v in props.items()}
    left_ips = left_ips[keep]
    right_ips = right_ips[keep]

    # Extend edges while signal is above baseline + threshold
    above = tic - baseline
    threshold = tic.max() * edge_threshold
    starts = np.zeros(len(peaks_idx), dtype=int)
    stops = np.zeros(len(peaks_idx), dtype=int)
    for i, pk in enumerate(peaks_idx):
        j = int(left_ips[i])
        while j > 0 and above[j] > threshold and pk - j < max_half_width:
            j -= 1
        starts[i] = j
        j = int(right_ips[i])
        while j < len(tic) - 1 and above[j] > threshold and j - pk < max_half_width:
            j += 1
        stops[i] = j

    # Resolve overlaps: split at valley between overlapping peaks
    for i in range(len(peaks_idx) - 1):
        if stops[i] > starts[i + 1]:
            valley = np.argmin(tic[peaks_idx[i]:peaks_idx[i + 1]]) + peaks_idx[i]
            stops[i] = valley
            starts[i + 1] = valley

    return [
        Peak(apex=int(pk), start=int(s), stop=int(e), prominence=float(props["prominences"][i]))
        for i, (pk, s, e) in enumerate(zip(peaks_idx, starts, stops))
    ]
