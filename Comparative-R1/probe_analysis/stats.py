from __future__ import annotations

from typing import Any

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch y_true={y_true.shape} y_pred={y_pred.shape}")

    labels = np.unique(y_true)
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []
    eps = 1e-12
    for c in labels:
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(per_class_recall) if per_class_recall else 0.0),
        "macro_f1": float(np.mean(per_class_f1) if per_class_f1 else 0.0),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    if n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {}

    all_scores: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }

    idx = np.arange(n)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(idx, size=n, replace=True)
        m = compute_metrics(y_true[sample_idx], y_pred[sample_idx])
        for k, v in m.items():
            all_scores[k].append(v)

    out: dict[str, dict[str, float]] = {}
    for k, vals in all_scores.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[k] = {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }
    return out


def json_ready(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_ready(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x
