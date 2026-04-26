from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = compute_midrank(positive_examples[r])
        ty[r] = compute_midrank(negative_examples[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_pvalue(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    order = np.argsort(-y_true)
    y_true = y_true[order]
    preds = np.vstack([score_a[order], score_b[order]])
    label_1_count = int(y_true.sum())
    aucs, covariance = fast_delong(preds, label_1_count)
    diff = float(aucs[0] - aucs[1])
    l = np.array([[1, -1]], dtype=float)
    variance = float(l @ covariance @ l.T)
    z = diff / np.sqrt(max(variance, 1e-12))
    p = 2.0 * norm.sf(abs(z))
    return {
        "auc_a": float(roc_auc_score(y_true, preds[0])),
        "auc_b": float(roc_auc_score(y_true, preds[1])),
        "auc_diff": diff,
        "z": float(z),
        "p_value_two_sided": float(p),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired DeLong test for comparing correlated AUROCs.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--y-true", default="y_true")
    parser.add_argument("--score-a", default="score_unaided")
    parser.add_argument("--score-b", default="score_ai")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    result = delong_pvalue(
        df[args.y_true].to_numpy(dtype=int),
        df[args.score_a].to_numpy(dtype=float),
        df[args.score_b].to_numpy(dtype=float),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
