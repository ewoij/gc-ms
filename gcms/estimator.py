"""Component count estimation using SVD features + RandomForest."""

from pathlib import Path

import joblib
import numpy as np

DEFAULT_MODEL = Path("models/component_counter/model.joblib")
NUM_SVD_FEATURES = 20


def estimate_components(M: np.ndarray, model_path: Path = DEFAULT_MODEL) -> int:
    """Estimate the number of components in an intensity matrix.

    Args:
        M: intensity matrix, shape (scans, mz)
        model_path: path to trained model

    Returns:
        Estimated number of components.
    """
    model = joblib.load(model_path)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if len(s) < NUM_SVD_FEATURES:
        s = np.pad(s, (0, NUM_SVD_FEATURES - len(s)))
    features = (s[:NUM_SVD_FEATURES] / s[0]).reshape(1, -1)
    return int(model.predict(features)[0])
