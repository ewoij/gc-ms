"""Train a component count estimator using SVD features.

Usage:
    uv run python tools/train_model.py --data data/synthetic_peaks
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


NUM_SVD_FEATURES = 20


def extract_features(ms: np.ndarray) -> np.ndarray:
    U, s, Vt = np.linalg.svd(ms, full_matrices=False)
    # Pad if fewer than NUM_SVD_FEATURES singular values
    if len(s) < NUM_SVD_FEATURES:
        s = np.pad(s, (0, NUM_SVD_FEATURES - len(s)))
    features = s[:NUM_SVD_FEATURES] / s[0]
    return features


def load_dataset(data_dir: Path):
    with open(data_dir / "metadata.csv") as f:
        rows = list(csv.DictReader(f))

    X, y = [], []
    for row in rows:
        ms_path = data_dir / row["sample_id"] / "ms.npy"
        ms = np.load(ms_path)
        X.append(extract_features(ms))
        y.append(int(row["num_components"]))

    return np.array(X), np.array(y)


def train(data_dir: Path, out_dir: Path):
    print("Loading dataset...")
    X, y = load_dataset(data_dir)
    print(f"  {len(X)} samples, {NUM_SVD_FEATURES} features, labels {y.min()}-{y.max()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print("Training...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {acc:.1%}")

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{classification_report(y_test, y_pred)}")
    print("Confusion matrix:")
    print(cm)

    # Feature importances
    print("\nTop SVD feature importances:")
    for i in np.argsort(clf.feature_importances_)[::-1][:5]:
        print(f"  s[{i}]: {clf.feature_importances_[i]:.3f}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "model.joblib")

    metrics = {
        "accuracy": acc,
        "num_train": len(X_train),
        "num_test": len(X_test),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  -> {out_dir}/model.joblib")
    print(f"  -> {out_dir}/metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Train component count estimator")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_peaks"))
    parser.add_argument("--outdir", type=Path, default=Path("models/component_counter"))
    args = parser.parse_args()
    train(args.data, args.outdir)


if __name__ == "__main__":
    main()
