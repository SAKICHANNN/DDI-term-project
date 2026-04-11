import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def build_feature_names(n_features: int) -> list[str]:
    if n_features % 2 == 0:
        half = n_features // 2
        left = [f"DrugA_F{i}" for i in range(half)]
        right = [f"DrugB_F{i}" for i in range(half)]
        return left + right
    return [f"Feature_{i}" for i in range(n_features)]


def extract_path_conditions(estimator, sample: np.ndarray, feature_names: list[str]) -> tuple[list[dict], int, np.ndarray]:
    node_indicator = estimator.decision_path(sample.reshape(1, -1))
    node_index = node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[1]]

    tree = estimator.tree_
    conditions: list[dict] = []
    for node_id in node_index:
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child == right_child:
            continue

        feature_idx = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        sample_value = float(sample[feature_idx])

        if sample_value <= threshold:
            operator = "<="
            next_node = int(left_child)
        else:
            operator = ">"
            next_node = int(right_child)

        conditions.append(
            {
                "node_id": int(node_id),
                "feature_index": feature_idx,
                "feature_name": feature_names[feature_idx],
                "operator": operator,
                "threshold": threshold,
                "sample_value": sample_value,
                "next_node": next_node,
            }
        )

    leaf_id = int(estimator.apply(sample.reshape(1, -1))[0])
    leaf_counts = tree.value[leaf_id][0]
    return conditions, leaf_id, leaf_counts


def explain_sample(rf_model, sample: np.ndarray, sample_index: int, feature_names: list[str], top_k: int, y_true: int | None):
    positive_class_index = int(np.where(rf_model.classes_ == 1)[0][0])

    tree_votes = []
    for tree_idx, estimator in enumerate(rf_model.estimators_):
        _, leaf_id, leaf_counts = extract_path_conditions(estimator, sample, feature_names)
        total = float(np.sum(leaf_counts))
        pos_prob = float(leaf_counts[positive_class_index] / total) if total > 0 else 0.0
        if pos_prob > 0.5:
            tree_votes.append(
                {
                    "tree_index": tree_idx,
                    "leaf_id": leaf_id,
                    "leaf_confidence": pos_prob,
                }
            )

    tree_votes.sort(key=lambda item: item["leaf_confidence"], reverse=True)
    top_votes = tree_votes[:top_k]

    ranked_paths = []
    for rank, vote in enumerate(top_votes, start=1):
        estimator = rf_model.estimators_[vote["tree_index"]]
        conditions, _, _ = extract_path_conditions(estimator, sample, feature_names)
        ranked_paths.append(
            {
                "rank": rank,
                "tree_index": vote["tree_index"],
                "leaf_confidence": vote["leaf_confidence"],
                "path_length": len(conditions),
                "conditions": conditions,
            }
        )

    sample_proba = float(rf_model.predict_proba(sample.reshape(1, -1))[0][positive_class_index])
    sample_pred = int(sample_proba >= 0.5)
    result = {
        "sample_index": sample_index,
        "y_pred": sample_pred,
        "y_pred_proba": sample_proba,
        "positive_voting_tree_count": len(tree_votes),
        "top_k": top_k,
        "ranked_paths": ranked_paths,
    }
    if y_true is not None:
        result["y_true"] = int(y_true)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TRDP explanations from a trained Random Forest model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../rf_model.pkl",
        help="Path to trained Random Forest pickle file.",
    )
    parser.add_argument(
        "--x-path",
        type=str,
        default="../X_test.npy",
        help="Path to feature matrix (NumPy .npy).",
    )
    parser.add_argument(
        "--y-path",
        type=str,
        default="../y_test.npy",
        help="Path to labels (NumPy .npy).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory for generated TRDP output files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top-ranked positive-voting trees to output per sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of samples to explain.",
    )
    parser.add_argument(
        "--only-positive-predictions",
        action="store_true",
        help="If set, only explain samples predicted as positive.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="",
        help="Optional comma-separated sample indices to explain (overrides selection rules).",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    model_path = resolve_path(script_dir, args.model_path)
    x_path = resolve_path(script_dir, args.x_path)
    y_path = resolve_path(script_dir, args.y_path)
    output_dir = resolve_path(script_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as handle:
        rf_model = pickle.load(handle)

    X = np.load(x_path)
    y = np.load(y_path) if y_path.exists() else None

    feature_names = build_feature_names(X.shape[1])
    positive_class_index = int(np.where(rf_model.classes_ == 1)[0][0])
    proba = rf_model.predict_proba(X)[:, positive_class_index]
    pred = (proba >= 0.5).astype(int)

    if args.sample_indices.strip():
        selected_indices = [
            int(x.strip()) for x in args.sample_indices.split(",") if x.strip()
        ]
    else:
        all_indices = np.arange(X.shape[0])
        if args.only_positive_predictions:
            all_indices = all_indices[pred == 1]
        selected_indices = all_indices[: args.max_samples].tolist()

    explanations = []
    for sample_index in selected_indices:
        y_true = int(y[sample_index]) if y is not None else None
        explanations.append(
            explain_sample(
                rf_model=rf_model,
                sample=X[sample_index],
                sample_index=sample_index,
                feature_names=feature_names,
                top_k=args.top_k,
                y_true=y_true,
            )
        )

    explanation_path = output_dir / "trdp_explanations.json"
    summary_path = output_dir / "trdp_summary.json"

    with open(explanation_path, "w", encoding="utf-8") as handle:
        json.dump(explanations, handle, indent=2)

    summary = {
        "model_path": str(model_path),
        "x_path": str(x_path),
        "y_path": str(y_path) if y_path.exists() else None,
        "n_features": int(X.shape[1]),
        "n_total_samples": int(X.shape[0]),
        "n_explained_samples": int(len(explanations)),
        "top_k": int(args.top_k),
        "only_positive_predictions": bool(args.only_positive_predictions),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved: {explanation_path}")
    print(f"Saved: {summary_path}")
    print(f"Explained samples: {len(explanations)}")


if __name__ == "__main__":
    main()
