from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_point_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
    return {"auroc": float(auroc), "sensitivity": float(sensitivity), "specificity": float(specificity)}


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    metrics = {"auroc": [], "sensitivity": [], "specificity": []}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        point = compute_point_metrics(yt, y_score[idx], y_pred[idx])
        for key in metrics:
            if np.isfinite(point[key]):
                metrics[key].append(point[key])
    summary: dict[str, dict[str, float]] = {}
    point = compute_point_metrics(y_true, y_score, y_pred)
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        summary[key] = {
            "value": float(point[key]),
            "ci_lower": float(np.nanpercentile(arr, 2.5)) if arr.size else float("nan"),
            "ci_upper": float(np.nanpercentile(arr, 97.5)) if arr.size else float("nan"),
            "n_bootstrap_effective": int(arr.size),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap 95% CIs for AUROC, sensitivity, and specificity.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--y-true", default="y_true")
    parser.add_argument("--y-score", default="y_score")
    parser.add_argument("--y-pred", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    y_true = df[args.y_true].to_numpy(dtype=int)
    y_score = df[args.y_score].to_numpy(dtype=float)
    y_pred = df[args.y_pred].to_numpy(dtype=int) if args.y_pred else (y_score >= args.threshold).astype(int)
    print(json.dumps(bootstrap_ci(y_true, y_score, y_pred, args.n_bootstrap, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
