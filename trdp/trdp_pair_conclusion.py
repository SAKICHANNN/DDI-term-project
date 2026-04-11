import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_selection import VarianceThreshold

from trdp_analysis import build_feature_names, explain_sample
from trdp_chain_report import condition_to_text, load_maccs_patterns
from maccs_reference_loader import load_maccs_human_reference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TRDP conflict conclusion for a specific DrugBank pair."
    )
    parser.add_argument("--drug-a-id", type=str, required=True, help="DrugBank ID for drug A")
    parser.add_argument("--drug-b-id", type=str, required=True, help="DrugBank ID for drug B")
    parser.add_argument("--model-path", type=str, default="../rf_model.pkl")
    parser.add_argument("--fingerprints-path", type=str, default="../drug_fingerprints.npz")
    parser.add_argument("--positive-samples-path", type=str, default="../positive_samples.npz")
    parser.add_argument("--negative-samples-path", type=str, default="../negative_samples.npz")
    parser.add_argument(
        "--maccs-reference-xlsx",
        type=str,
        default="../MACCS_Keys_Human_Readable_Reference(1).xlsx",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: str):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_selector_mask(positive_samples_path: Path, negative_samples_path: Path):
    pos = np.load(positive_samples_path)
    neg = np.load(negative_samples_path)
    X = np.vstack([pos["X"], neg["X"]])
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X)
    return selector.get_support()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    model_path = resolve_path(script_dir, args.model_path)
    fingerprints_path = resolve_path(script_dir, args.fingerprints_path)
    positive_samples_path = resolve_path(script_dir, args.positive_samples_path)
    negative_samples_path = resolve_path(script_dir, args.negative_samples_path)
    maccs_reference_xlsx = resolve_path(script_dir, args.maccs_reference_xlsx)
    output_dir = resolve_path(script_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as handle:
        rf_model = pickle.load(handle)

    fp_data = np.load(fingerprints_path, allow_pickle=True)
    fingerprints = dict(zip(fp_data["ids"], fp_data["fps"]))

    if args.drug_a_id not in fingerprints:
        raise ValueError(f"Drug A id not found in fingerprints: {args.drug_a_id}")
    if args.drug_b_id not in fingerprints:
        raise ValueError(f"Drug B id not found in fingerprints: {args.drug_b_id}")

    raw_feature = np.concatenate([fingerprints[args.drug_a_id], fingerprints[args.drug_b_id]]).astype(float)

    model_dim = int(rf_model.n_features_in_)
    if raw_feature.shape[0] == model_dim:
        final_feature = raw_feature
        mapping_note = "Model feature dimension matches raw concatenated vector."
    else:
        mask = load_selector_mask(positive_samples_path, negative_samples_path)
        reduced_feature = raw_feature[mask]
        if reduced_feature.shape[0] != model_dim:
            raise ValueError(
                f"Feature mismatch after mask transform. model_dim={model_dim}, "
                f"reduced_dim={reduced_feature.shape[0]}"
            )
        final_feature = reduced_feature
        mapping_note = (
            "Applied variance-threshold mask inferred from positive/negative sample files. "
            "MACCS key mapping confidence may be reduced if training-time mask differs."
        )

    feature_names = build_feature_names(model_dim)
    explanation = explain_sample(
        rf_model=rf_model,
        sample=final_feature,
        sample_index=-1,
        feature_names=feature_names,
        top_k=args.top_k,
        y_true=None,
    )

    patterns = load_maccs_patterns()
    human_reference = load_maccs_human_reference(maccs_reference_xlsx)
    enriched_chains = []
    for chain in explanation["ranked_paths"]:
        enriched_steps = [
            condition_to_text(
                condition,
                model_dim,
                patterns,
                maccs_human_reference=human_reference,
            )
            for condition in chain["conditions"]
        ]
        enriched_chains.append(
            {
                "rank": chain["rank"],
                "tree_index": chain["tree_index"],
                "leaf_confidence": chain["leaf_confidence"],
                "path_length": chain["path_length"],
                "chain_steps": enriched_steps,
            }
        )

    result = {
        "drug_a_id": args.drug_a_id,
        "drug_b_id": args.drug_b_id,
        "y_pred": explanation["y_pred"],
        "y_pred_proba": explanation["y_pred_proba"],
        "positive_voting_tree_count": explanation["positive_voting_tree_count"],
        "top_k": args.top_k,
        "mapping_note": mapping_note,
        "chains": enriched_chains,
    }

    stem = f"{args.drug_a_id}_{args.drug_b_id}"
    json_path = output_dir / f"pair_conclusion_{stem}.json"
    txt_path = output_dir / f"pair_conclusion_{stem}.txt"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    lines = []
    lines.append(
        f"{args.drug_a_id} and {args.drug_b_id} are predicted to "
        f"{'conflict' if result['y_pred'] == 1 else 'not conflict'} "
        f"(probability={result['y_pred_proba']:.3f}, positive_voting_trees={result['positive_voting_tree_count']})."
    )
    lines.append("Reasoning chain from top-ranked tree paths:")
    for chain in result["chains"]:
        lines.append(
            f"- Rank {chain['rank']} | tree={chain['tree_index']} "
            f"| leaf_confidence={chain['leaf_confidence']:.3f}"
        )
        for idx, step in enumerate(chain["chain_steps"][:6], start=1):
            readable_phrase = step["substructure_readable"].strip()
            if readable_phrase.lower().startswith("contains "):
                readable_phrase = readable_phrase[9:]
            if readable_phrase:
                readable_phrase = readable_phrase[0].lower() + readable_phrase[1:]

            base = step["interpretation"]
            if "contains this substructure" in base:
                readable_interpretation = base.replace(
                    "contains this substructure", f"contains {readable_phrase}"
                )
            elif "does not contain this substructure" in base:
                readable_interpretation = base.replace(
                    "does not contain this substructure",
                    f"does not contain {readable_phrase}",
                )
            else:
                readable_interpretation = base.replace("this substructure", readable_phrase)
            lines.append(
                f"  ({idx}) {readable_interpretation} "
                f"(feature={step['feature_name']}, MACCS={step['maccs_key']})"
            )
    lines.append("")
    lines.append("Note: This is a model-inferred hypothesis, not a laboratory-confirmed mechanism.")
    lines.append(f"Mapping note: {mapping_note}")

    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
