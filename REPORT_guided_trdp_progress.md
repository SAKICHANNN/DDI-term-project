# Guided TRDP Progress Report

## Date

2026-04-11

## Scope

This report summarizes the current implementation status of the two-stage guided TRDP plan:

1. Stage-1 baseline Random Forest with SHAP importance extraction.
2. Stage-2 guided forest with SHAP-informed feature sampling.
3. Baseline vs guided comparison on predictive and interpretability metrics.

---

## What has been implemented

### 1) Project structure refactor

- Core scripts were moved from repository root into structured folders:
  - `pipelines/data/`
  - `pipelines/models/`
  - `pipelines/explainability/`
  - `pipelines/experiments/guided_trdp/`
- Path management is centralized in `project_paths.py`.
- Legacy root outputs were moved into `artifacts/legacy_root_outputs/`.

### 2) Guided TRDP experiment code

New files:

- `pipelines/experiments/guided_trdp/guided_forest.py`
  - Lightweight guided forest implementation.
  - Uses weighted per-tree feature-pool sampling.
- `pipelines/experiments/guided_trdp/run_guided_trdp_experiment.py`
  - End-to-end experiment runner:
    - trains baseline RF,
    - computes stage-1 SHAP importances,
    - trains guided RF for configured alpha values,
    - computes predictive metrics and interpretability alignment,
    - exports CSV/JSON/Markdown artifacts.

### 3) Evaluation outputs generated

Output directories:

- `artifacts/explanations/guided_trdp_concat_324_k1_n10/`
- `artifacts/explanations/guided_trdp_concat_324_k1_n20/`
- `artifacts/explanations/guided_trdp_concat_324_k3_n10/`
- `artifacts/explanations/guided_trdp_concat_324_k3_n20/`
- `artifacts/explanations/guided_trdp_add_167_k1_n10/`
- `artifacts/explanations/guided_trdp_add_167_k1_n20/`
- `artifacts/explanations/guided_trdp_add_167_k3_n10/`
- `artifacts/explanations/guided_trdp_add_167_k3_n20/`
- merged summary:
  - `artifacts/explanations/guided_trdp_expanded/all_runs_metrics.csv`
  - `artifacts/explanations/guided_trdp_expanded/aggregated_mean_std.csv`

Generated files include:

- `comparison_metrics.csv`
- `interpretability_comparison.md`
- `guided_sampling_config.json`
- `stage1_shap_importance_seed_42.json` / `52` / `62`
- baseline and guided model pickles for tested settings

---

## Expanded run configuration (not only top-1)

Core matrix:

- seeds: `42, 52, 62`
- alphas: `0.0, 0.3, 0.7, 1.0`
- top_k: `1, 3`
- top_n: `10, 20`
- shap_max_samples: `150`

Run policy:

- Both representations were run with the same matrix:
  - 324-dim (`concat` + variance filtering to 324)
  - 167-dim (`additive`)
- This update explicitly includes `top_k=3` in addition to `top_k=1`.

---

## Aggregated results (mean over seeds 42/52/62)

### A) 324-dim representation (concat, filtered to 324)

Best guided setting by Jaccard in all four slices is `alpha=1.0`.

- `top_k=1, top_n=10`: Jaccard `0.0778 -> 0.1617` (`+0.0840`), AUC `-0.0118`
- `top_k=1, top_n=20`: Jaccard `0.1043 -> 0.1687` (`+0.0644`), AUC `-0.0118`
- `top_k=3, top_n=10`: Jaccard `0.0903 -> 0.1378` (`+0.0476`), AUC `-0.0118`
- `top_k=3, top_n=20`: Jaccard `0.1354 -> 0.2170` (`+0.0816`), AUC `-0.0118`

### B) 167-dim representation (additive)

Best guided setting by Jaccard in all four slices is also `alpha=1.0`.

- `top_k=1, top_n=10`: Jaccard `0.1217 -> 0.1771` (`+0.0554`), AUC `-0.0114`
- `top_k=1, top_n=20`: Jaccard `0.1549 -> 0.2105` (`+0.0555`), AUC `-0.0114`
- `top_k=3, top_n=10`: Jaccard `0.1279 -> 0.1756` (`+0.0477`), AUC `-0.0114`
- `top_k=3, top_n=20`: Jaccard `0.1965 -> 0.2742` (`+0.0776`), AUC `-0.0114`

### Main observation

- 324-dim and 167-dim both run end-to-end with the same guided TRDP framework.
- After multi-seed aggregation, guided sampling improves TRDP-SHAP overlap in all tested `top_k/top_n` slices for both representations.
- The overlap gain is consistently accompanied by a moderate predictive drop (about `0.011` AUC in these runs), indicating a stable interpretability-performance trade-off.
- Conclusion is no longer based on only top-1; top-3 runs show the same trend direction.

---

## Metric definitions (protocol-level)

This section defines the exact sets and formulas used in all tables.

### 1) Set definitions for TRDP-SHAP overlap

For each positive test sample `x`:

- `S_trdp(x)`:
  - union of feature names appearing on the selected TRDP decision paths.
  - if `top_k=1`: use only the highest-ranked path.
  - if `top_k=3`: use the union across the top-3 ranked paths.
- `S_shap`:
  - global SHAP top-`N` feature set from stage-1 baseline RF, where `N=top_n`.
  - SHAP importances are computed as mean absolute SHAP values on a validation subset.

Jaccard at sample level:

- `J(x) = |S_trdp(x) ∩ S_shap| / |S_trdp(x) ∪ S_shap|`

Reported overlap metric:

- `jaccard_mean = mean_x J(x)` over selected positive test samples.

Selected samples rule:

- positive prediction set = indices where `y_prob >= 0.5`
- evaluate first `20` such samples (or fewer if unavailable)

### 2) Binary classification metrics

Given confusion matrix counts (`TP`, `TN`, `FP`, `FN`):

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2 * Precision * Recall / (Precision + Recall)`
- `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- `AUC`:
  - ROC-AUC computed from true labels and predicted probabilities.

Prediction threshold:

- binary prediction uses threshold `0.5` on predicted probability.

### 3) Path readability metric

- `path_length` for one decision path:
  - number of non-leaf split conditions on that path.
- For one sample with `top_k > 1`:
  - average of selected path lengths.
- Reported `path_length_mean`:
  - mean over evaluated positive samples.

### 4) Aggregation across runs

- Raw per-run metrics are stored in:
  - `artifacts/explanations/guided_trdp_expanded/all_runs_metrics.csv`
- Final summary table is:
  - `artifacts/explanations/guided_trdp_expanded/aggregated_mean_std.csv`
- Aggregation keys:
  - `representation, top_k, top_n, model, alpha`
- For each metric, report:
  - `mean` and `std` over seeds (`42, 52, 62`).

### 5) Model settings used for this report

- Baseline model:
  - standard RF (`alpha=0.0` control row).
- Guided model:
  - SHAP-guided weighted feature-pool sampling with
    - `p_i ∝ (importance_i + epsilon)^alpha`
    - `epsilon = 1e-8`
    - `alpha in {0.3, 0.7, 1.0}`
- Shared settings:
  - `n_estimators=100`
  - classification threshold `0.5`
  - seeds `42, 52, 62`.

---

## Current limitations

1. Guided forest still uses weighted per-tree feature-pool sampling (not native sklearn weighted split-candidate sampling).
2. This report is aggregated at metric level; 3-5 case-level chemistry narratives still need final curation for the LaTeX appendix.
3. If needed, we can still extend to additional seeds for tighter variance estimates.

---

## Next steps

1. Extract and format 3-5 case-level explanation outputs for report insertion.
2. Add one compact table in `CS3264_Final_Report.tex` using `aggregated_mean_std.csv`.
3. Align method text with the final setting:
   - top-1 as main chain view,
   - top-3 as robustness check,
   - both 324 and 167 representations included.

---

## Status

- [x] Infrastructure ready
- [x] Baseline + guided runner ready
- [x] Expanded matrix complete (both 324-dim and 167-dim; top_k=1/3; top_n=10/20; seeds=42/52/62)
- [x] Aggregated mean/std exported
- [ ] Case-level report snippets finalized
- [ ] Final LaTeX tables finalized
