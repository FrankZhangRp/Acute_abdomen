from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nonparametric tests used in the paper.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    wil = subparsers.add_parser("wilcoxon", help="Paired Wilcoxon signed-rank test.")
    wil.add_argument("--csv", required=True)
    wil.add_argument("--x-col", required=True)
    wil.add_argument("--y-col", required=True)
    wil.add_argument("--zero-method", default="wilcox", choices=["wilcox", "pratt", "zsplit"])
    mwu = subparsers.add_parser("mannwhitney", help="Independent Mann-Whitney U test.")
    mwu.add_argument("--csv", required=True)
    mwu.add_argument("--group-col", required=True)
    mwu.add_argument("--value-col", required=True)
    mwu.add_argument("--group-a", required=True)
    mwu.add_argument("--group-b", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.csv)
    if args.command == "wilcoxon":
        x = df[args.x_col].to_numpy(dtype=float)
        y = df[args.y_col].to_numpy(dtype=float)
        stat = wilcoxon(x, y, zero_method=args.zero_method, alternative="two-sided")
        payload = {
            "test": "wilcoxon_signed_rank",
            "n_pairs": int(len(df)),
            "median_x": float(np.nanmedian(x)),
            "median_y": float(np.nanmedian(y)),
            "statistic": float(stat.statistic),
            "p_value_two_sided": float(stat.pvalue),
        }
    else:
        group_a = df.loc[df[args.group_col] == args.group_a, args.value_col].to_numpy(dtype=float)
        group_b = df.loc[df[args.group_col] == args.group_b, args.value_col].to_numpy(dtype=float)
        stat = mannwhitneyu(group_a, group_b, alternative="two-sided")
        payload = {
            "test": "mann_whitney_u",
            "n_group_a": int(len(group_a)),
            "n_group_b": int(len(group_b)),
            "median_group_a": float(np.nanmedian(group_a)),
            "median_group_b": float(np.nanmedian(group_b)),
            "statistic": float(stat.statistic),
            "p_value_two_sided": float(stat.pvalue),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
