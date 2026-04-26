# Statistical Analysis Utilities

This folder contains reusable analysis code matching the paper's statistical-analysis protocol.

## Protocol implemented

- Python version: **3.10.14**
- Significance threshold: **two-sided P < 0.05**
- 95% confidence intervals for model metrics: **10,000 bootstrap iterations**
- Reader-study accuracy effect of AI: **generalized linear mixed-effects model (GLMM)**
- AUROC comparison for paired unaided vs AI-assisted sessions: **paired DeLong test**
- Paired reading-time comparison: **Wilcoxon signed-rank test**
- Independent workflow-turnaround-time comparison: **Mann-Whitney U test**

## Files

- `bootstrap_ci.py`
  - Bootstrap AUROC / sensitivity / specificity with percentile CIs.
- `delong.py`
  - Paired DeLong test for correlated ROC curves.
- `mrmc_glmm.py`
  - Logistic mixed-effects analysis for multi-reader multi-case (MRMC) studies.
- `nonparametric_tests.py`
  - Wilcoxon signed-rank and Mann-Whitney U wrappers.
- `requirements.txt`
  - Minimal package list for this folder.

## Installation

```bash
pip install -r statistics/requirements.txt
```

## Expected CSV schemas

### 1) Bootstrap metrics

CSV columns:

- `y_true`: binary ground-truth label (`0/1`)
- `y_score`: model score / probability
- optional `y_pred`: binary prediction (`0/1`)

If `y_pred` is absent, pass `--threshold` and the script will derive predictions from `y_score`.

```bash
python statistics/bootstrap_ci.py \
  --csv predictions.csv \
  --y-true y_true \
  --y-score y_score \
  --threshold 0.5 \
  --n-bootstrap 10000
```

### 2) Paired DeLong test

CSV columns:

- `y_true`
- `score_unaided`
- `score_ai`

```bash
python statistics/delong.py \
  --csv delong_input.csv \
  --y-true y_true \
  --score-a score_unaided \
  --score-b score_ai
```

### 3) MRMC GLMM

CSV columns:

- `reader_id`
- `case_id`
- `ai_assisted` (`0/1`)
- `correct` (`0/1`)

```bash
python statistics/mrmc_glmm.py \
  --csv mrmc.csv \
  --reader-col reader_id \
  --case-col case_id \
  --ai-col ai_assisted \
  --target-col correct
```

### 4) Paired reading times

CSV columns:

- `reader_id`
- `case_id`
- `time_unaided`
- `time_ai`

```bash
python statistics/nonparametric_tests.py wilcoxon \
  --csv reading_times.csv \
  --x-col time_unaided \
  --y-col time_ai
```

### 5) Workflow turnaround times

CSV columns:

- `cohort`
- `tat_minutes`

```bash
python statistics/nonparametric_tests.py mannwhitney \
  --csv workflow.csv \
  --group-col cohort \
  --value-col tat_minutes \
  --group-a historical \
  --group-b reconstructed
```
