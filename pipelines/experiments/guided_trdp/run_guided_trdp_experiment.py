from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

from project_paths import EXPLANATIONS_DIR, X_TEST_NPY, X_TRAIN_NPY, X_VAL_NPY, Y_TEST_NPY, Y_TRAIN_NPY
from trdp.trdp_analysis import build_feature_names, explain_sample
from guided_forest import GuidedRandomForest


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def stage1_shap_importance(model: RandomForestClassifier, X_val: np.ndarray) -> np.ndarray:
    sample = X_val[: min(300, len(X_val))]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
    else:
        shap_pos = shap_values[:, :, 1]
    return np.abs(shap_pos).mean(axis=0)


def compute_weighted_probs(importances: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
    weights = np.power(importances + epsilon, alpha)
    return weights / weights.sum()


def _guided_path_features(model: GuidedRandomForest, sample: np.ndarray, top_k: int, feature_names: list[str]):
    tree_votes = []
    for idx, tree in enumerate(model.trees_):
        local = sample[tree.feature_indices].reshape(1, -1)
        leaf_id = int(tree.estimator.apply(local)[0])
        leaf_counts = tree.estimator.tree_.value[leaf_id][0]
        total = float(np.sum(leaf_counts))
        pos_prob = float(leaf_counts[1] / total) if total > 0 and len(leaf_counts) > 1 else 0.0
        if pos_prob > 0.5:
            tree_votes.append((idx, pos_prob))
    tree_votes.sort(key=lambda x: x[1], reverse=True)

    features = set()
    path_lengths = []
    for idx, _ in tree_votes[:top_k]:
        tree = model.trees_[idx]
        local = sample[tree.feature_indices].reshape(1, -1)
        node_indicator = tree.estimator.decision_path(local)
        node_index = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
        length = 0
        for node_id in node_index:
            left = tree.estimator.tree_.children_left[node_id]
            right = tree.estimator.tree_.children_right[node_id]
            if left == right:
                continue
            local_feature_idx = int(tree.estimator.tree_.feature[node_id])
            global_idx = int(tree.feature_indices[local_feature_idx])
            features.add(feature_names[global_idx])
            length += 1
        path_lengths.append(length)
    return features, path_lengths


def compute_interpretability_stats(
    model_name: str,
    model,
    X_test: np.ndarray,
    y_prob: np.ndarray,
    top_k: int,
    shap_top_features: set[str],
) -> dict:
    feature_names = build_feature_names(X_test.shape[1])
    positive_indices = np.where(y_prob >= 0.5)[0][:20]

    jaccards = []
    avg_lengths = []

    for idx in positive_indices:
        x = X_test[idx]
        if model_name == "baseline":
            exp = explain_sample(model, x, int(idx), feature_names, top_k=top_k, y_true=None)
            path_features = set()
            lengths = []
            for path in exp["ranked_paths"]:
                lengths.append(path["path_length"])
                for cond in path["conditions"]:
                    path_features.add(cond["feature_name"])
        else:
            path_features, lengths = _guided_path_features(model, x, top_k=top_k, feature_names=feature_names)

        if path_features:
            inter = len(path_features & shap_top_features)
            union = len(path_features | shap_top_features)
            jaccards.append(inter / union if union else 0.0)
        if lengths:
            avg_lengths.append(float(np.mean(lengths)))

    return {
        "samples_used": int(len(positive_indices)),
        "jaccard_mean": float(np.mean(jaccards)) if jaccards else 0.0,
        "path_length_mean": float(np.mean(avg_lengths)) if avg_lengths else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline-vs-guided TRDP experiment.")
    parser.add_argument("--seeds", type=str, default="42,52,62", help="Comma-separated seeds.")
    parser.add_argument("--alphas", type=str, default="0,0.3,0.7,1.0", help="Comma-separated alphas.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--shap-max-samples", type=int, default=300)
    parser.add_argument("--run-name", type=str, default="guided_trdp", help="Output subfolder name under artifacts/explanations/")
    args = parser.parse_args()

    out_dir = EXPLANATIONS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(X_TRAIN_NPY)
    y_train = np.load(Y_TRAIN_NPY)
    X_val = np.load(X_VAL_NPY)
    X_test = np.load(X_TEST_NPY)
    y_test = np.load(Y_TEST_NPY)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    epsilon = 1e-8
    top_k = args.top_k
    top_n = args.top_n

    def stage1_shap_importance_with_limit(model: RandomForestClassifier, X_val_: np.ndarray) -> np.ndarray:
        sample = X_val_[: min(args.shap_max_samples, len(X_val_))]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_pos = shap_values[1]
        else:
            shap_pos = shap_values[:, :, 1]
        return np.abs(shap_pos).mean(axis=0)

    rows = []
    for seed in seeds:
        print(f"[seed={seed}] Training baseline RF...")
        baseline = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        baseline.fit(X_train, y_train)

        y_prob_base = baseline.predict_proba(X_test)[:, 1]
        y_pred_base = (y_prob_base >= 0.5).astype(int)
        base_metrics = evaluate_binary(y_test, y_pred_base, y_prob_base)

        print(f"[seed={seed}] Computing SHAP importances...")
        importances = stage1_shap_importance_with_limit(baseline, X_val)
        importance_payload = {
            "seed": seed,
            "importance_mean_abs_shap": importances.tolist(),
        }
        with open(out_dir / f"stage1_shap_importance_seed_{seed}.json", "w", encoding="utf-8") as handle:
            json.dump(importance_payload, handle, indent=2)

        top_feature_names = build_feature_names(X_train.shape[1])
        top_indices = np.argsort(importances)[::-1][:top_n]
        shap_top_features = {top_feature_names[int(i)] for i in top_indices}

        base_interp = compute_interpretability_stats(
            model_name="baseline",
            model=baseline,
            X_test=X_test,
            y_prob=y_prob_base,
            top_k=top_k,
            shap_top_features=shap_top_features,
        )

        rows.append(
            {
                "seed": seed,
                "alpha": 0.0,
                "model": "baseline_rf",
                **base_metrics,
                **base_interp,
            }
        )

        with open(out_dir / f"rf_stage1_model_seed_{seed}.pkl", "wb") as handle:
            pickle.dump(baseline, handle)

        for alpha in [a for a in alphas if a > 0]:
            print(f"[seed={seed}, alpha={alpha}] Training guided RF...")
            probs = compute_weighted_probs(importances, alpha=alpha, epsilon=epsilon)
            guided = GuidedRandomForest(
                n_estimators=100,
                tree_feature_pool_size=int(np.sqrt(X_train.shape[1]) * 3),
                max_depth=None,
                min_samples_leaf=1,
                bootstrap=True,
                random_state=seed,
            )
            guided.fit(X_train, y_train, feature_probabilities=probs)
            y_prob_guided = guided.predict_proba(X_test)[:, 1]
            y_pred_guided = (y_prob_guided >= 0.5).astype(int)
            guided_metrics = evaluate_binary(y_test, y_pred_guided, y_prob_guided)
            guided_interp = compute_interpretability_stats(
                model_name="guided",
                model=guided,
                X_test=X_test,
                y_prob=y_prob_guided,
                top_k=top_k,
                shap_top_features=shap_top_features,
            )

            rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "model": "guided_rf",
                    **guided_metrics,
                    **guided_interp,
                }
            )

            with open(out_dir / f"rf_stage2_guided_model_seed_{seed}_alpha_{alpha}.pkl", "wb") as handle:
                pickle.dump(guided, handle)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_metrics.csv", index=False)

    config = {
        "alphas": alphas,
        "epsilon": epsilon,
        "seeds": seeds,
        "top_k": top_k,
        "top_n": top_n,
        "tree_feature_pool_size_formula": "int(sqrt(d)*3)",
        "note": "alpha=0 row is baseline RF; guided model uses weighted per-tree feature pool.",
    }
    with open(out_dir / "guided_sampling_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    summary_lines = [
        "# Interpretability Comparison",
        "",
        "This report compares baseline RF (uniform split-candidate sampling) and guided RF.",
        "Guided RF uses SHAP-derived feature probabilities for per-tree feature pool sampling.",
        "",
        "## Aggregate by model and alpha",
        "",
    ]
    agg = (
        df.groupby(["model", "alpha"], as_index=False)[
            ["auc", "f1", "precision", "recall", "accuracy", "jaccard_mean", "path_length_mean"]
        ]
        .mean()
        .sort_values(["model", "alpha"])
    )
    summary_lines.append(agg.to_csv(index=False))
    summary_lines.append("")
    summary_lines.append("## Key readouts")
    best_j = agg.sort_values("jaccard_mean", ascending=False).iloc[0]
    summary_lines.append(
        f"- Best Jaccard setting: {best_j['model']} alpha={best_j['alpha']} "
        f"(jaccard_mean={best_j['jaccard_mean']:.4f})"
    )
    summary_lines.append(
        "- Compare this against baseline_auc/f1 to determine the interpretability-performance trade-off."
    )

    (out_dir / "interpretability_comparison.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved experiment outputs to: {out_dir}")


if __name__ == "__main__":
    main()
