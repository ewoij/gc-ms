"""MCR-ALS (Multivariate Curve Resolution - Alternating Least Squares)."""

import numpy as np


def _enforce_unimodality(C):
    """Enforce unimodality on each column of C (each elution profile)."""
    for j in range(C.shape[1]):
        col = C[:, j]
        peak = np.argmax(col)

        # Before peak: enforce monotonic increase
        for i in range(peak - 1, -1, -1):
            if col[i] > col[i + 1]:
                col[i] = col[i + 1]

        # After peak: enforce monotonic decrease
        for i in range(peak + 1, len(col)):
            if col[i] > col[i - 1]:
                col[i] = col[i - 1]

    return C


def mcr_als(M, n_components, max_iter=100, tol=1e-6):
    """Run MCR-ALS on an intensity matrix.

    Args:
        M: intensity matrix, shape (scans, mz)
        n_components: number of components to resolve
        max_iter: maximum ALS iterations
        tol: convergence tolerance (relative change in residual)

    Returns:
        C: elution profiles, shape (scans, n_components)
        S: spectra, shape (n_components, mz)
    """
    M = M.astype(np.float64)
    n_scans, n_mz = M.shape

    # Initial guess: SVD
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    S = np.abs(Vt[:n_components])  # (n_components, mz)

    norm_M = np.linalg.norm(M)
    prev_residual = np.inf

    for iteration in range(max_iter):
        # Solve for C (profiles) given S
        # M ≈ C @ S → C = M @ S.T @ inv(S @ S.T)
        StS = S @ S.T
        try:
            C = M @ S.T @ np.linalg.inv(StS)
        except np.linalg.LinAlgError:
            C = M @ S.T @ np.linalg.pinv(StS)
        np.clip(C, 0, None, out=C)

        # Unimodality constraint
        C = _enforce_unimodality(C)

        # Solve for S (spectra) given C
        # M ≈ C @ S → S = inv(C.T @ C) @ C.T @ M
        CtC = C.T @ C
        try:
            S = np.linalg.solve(CtC, C.T @ M)
        except np.linalg.LinAlgError:
            S = np.linalg.pinv(CtC) @ C.T @ M
        np.clip(S, 0, None, out=S)

        # Check convergence
        residual = np.linalg.norm(M - C @ S) / norm_M
        change = abs(prev_residual - residual)
        if change < tol:
            break
        prev_residual = residual

    return C, S
