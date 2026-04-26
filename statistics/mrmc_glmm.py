from __future__ import annotations

import argparse
import json
import math

import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRMC GLMM for AI effect on diagnostic accuracy.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--reader-col", default="reader_id")
    parser.add_argument("--case-col", default="case_id")
    parser.add_argument("--ai-col", default="ai_assisted")
    parser.add_argument("--target-col", default="correct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv).copy()
    df[args.ai_col] = df[args.ai_col].astype(int)
    df[args.target_col] = df[args.target_col].astype(int)
    model = BinomialBayesMixedGLM.from_formula(
        f"{args.target_col} ~ {args.ai_col}",
        {"reader": f"0 + C({args.reader_col})", "case": f"0 + C({args.case_col})"},
        df,
    )
    result = model.fit_vb()
    fe_names = list(result.model.exog_names)
    ai_idx = fe_names.index(args.ai_col)
    coef = float(result.fe_mean[ai_idx])
    sd = float(result.fe_sd[ai_idx])
    z = coef / max(sd, 1e-12)
    p_value = 2.0 * norm.sf(abs(z))
    ci_low = coef - 1.96 * sd
    ci_high = coef + 1.96 * sd
    payload = {
        "fixed_effects": {name: float(val) for name, val in zip(fe_names, result.fe_mean)},
        "ai_effect_log_odds": coef,
        "ai_effect_log_odds_ci95": [ci_low, ci_high],
        "ai_effect_odds_ratio": math.exp(coef),
        "ai_effect_odds_ratio_ci95": [math.exp(ci_low), math.exp(ci_high)],
        "z_approx": z,
        "p_value_two_sided_approx": float(p_value),
        "note": "Python GLMM implementation with reader and case random intercepts, aligned to the paper's MRMC analysis description.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
