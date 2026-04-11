import argparse
import json
import re
from pathlib import Path

from rdkit.Chem import MACCSkeys
from maccs_reference_loader import load_maccs_human_reference
from substructure_text import smarts_to_human_text


def load_trdp_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_maccs_patterns() -> dict[int, str]:
    patterns = {}
    for key_id, value in MACCSkeys.smartsPatts.items():
        smarts = value[0]
        patterns[int(key_id)] = smarts
    return patterns


def parse_feature_name(feature_name: str):
    match_ab = re.match(r"^(Drug[AB])_F(\d+)$", feature_name)
    if match_ab:
        return {"scope": match_ab.group(1), "local_index": int(match_ab.group(2))}

    match_generic = re.match(r"^Feature_(\d+)$", feature_name)
    if match_generic:
        return {"scope": "Pair", "local_index": int(match_generic.group(1))}

    return {"scope": "Unknown", "local_index": None}


def resolve_maccs_key(local_index: int | None, total_features: int, feature_name: str):
    if local_index is None:
        return {
            "maccs_key": None,
            "mapping_confidence": "none",
            "mapping_note": f"Unrecognized feature name format: {feature_name}",
        }

    if total_features == 167:
        return {
            "maccs_key": local_index,
            "mapping_confidence": "high",
            "mapping_note": "Direct mapping for additive 167-dimensional representation.",
        }

    if total_features % 2 == 0 and feature_name.startswith(("DrugA_F", "DrugB_F")):
        return {
            "maccs_key": local_index,
            "mapping_confidence": "low",
            "mapping_note": (
                "Estimated mapping in reduced feature space; variance filtering may shift "
                "original MACCS key positions."
            ),
        }

    return {
        "maccs_key": local_index,
        "mapping_confidence": "low",
        "mapping_note": "Fallback index mapping; verify against original feature construction.",
    }


def condition_to_text(
    condition: dict,
    total_features: int,
    maccs_patterns: dict[int, str],
    maccs_human_reference: dict[int, dict] | None = None,
):
    parsed = parse_feature_name(condition["feature_name"])
    mapping = resolve_maccs_key(parsed["local_index"], total_features, condition["feature_name"])
    key_id = mapping["maccs_key"]
    smarts = maccs_patterns.get(key_id, "?") if key_id is not None else "?"
    reference = (maccs_human_reference or {}).get(key_id, {}) if key_id is not None else {}
    ref_label = reference.get("short_label", "")
    ref_desc = reference.get("description", "")
    if ref_desc:
        readable = ref_desc
        readable_source = "xlsx_reference"
    else:
        readable = smarts_to_human_text(smarts)
        readable_source = "heuristic_from_smarts"

    operator = condition["operator"]
    threshold = float(condition["threshold"])
    sample_value = float(condition["sample_value"])
    scope = parsed["scope"]

    if operator == ">" and abs(threshold - 0.5) < 1e-6:
        interpretation = f"{scope} contains this substructure"
    elif operator == "<=" and abs(threshold - 0.5) < 1e-6:
        interpretation = f"{scope} does not contain this substructure"
    elif operator == ">" and abs(threshold - 1.5) < 1e-6:
        interpretation = f"{scope} strongly indicates shared substructure signal"
    else:
        interpretation = f"{scope} satisfies split condition ({operator} {threshold:.3f})"

    return {
        "node_id": condition["node_id"],
        "feature_name": condition["feature_name"],
        "feature_value": sample_value,
        "split": f"{condition['feature_name']} {operator} {threshold:.3f}",
        "maccs_key": key_id,
        "smarts_pattern": smarts,
        "substructure_label": ref_label,
        "substructure_readable": readable,
        "substructure_readable_source": readable_source,
        "mapping_confidence": mapping["mapping_confidence"],
        "mapping_note": mapping["mapping_note"],
        "interpretation": interpretation,
    }


def explain_to_chain(
    explanations: list[dict],
    total_features: int,
    maccs_human_reference: dict[int, dict] | None = None,
):
    maccs_patterns = load_maccs_patterns()
    chain_output = []

    for sample in explanations:
        sample_entry = {
            "sample_index": sample["sample_index"],
            "y_true": sample.get("y_true"),
            "y_pred": sample["y_pred"],
            "y_pred_proba": sample["y_pred_proba"],
            "positive_voting_tree_count": sample["positive_voting_tree_count"],
            "chains": [],
        }

        for ranked in sample["ranked_paths"]:
            chain_steps = []
            for condition in ranked["conditions"]:
                chain_steps.append(
                    condition_to_text(
                        condition,
                        total_features,
                        maccs_patterns,
                        maccs_human_reference=maccs_human_reference,
                    )
                )

            sample_entry["chains"].append(
                {
                    "rank": ranked["rank"],
                    "tree_index": ranked["tree_index"],
                    "leaf_confidence": ranked["leaf_confidence"],
                    "path_length": ranked["path_length"],
                    "chain_steps": chain_steps,
                }
            )

        chain_output.append(sample_entry)

    return chain_output


def chain_to_text_report(chains: list[dict], total_features: int):
    lines = []
    lines.append("TRDP Chemical Chain Report")
    lines.append("=" * 80)
    lines.append(f"Model feature dimension: {total_features}")
    lines.append(
        "Note: This is a model-inferred substructure rule chain, not a verified biochemical mechanism."
    )
    lines.append("")

    for sample in chains:
        lines.append(
            f"[Sample {sample['sample_index']}] y_pred={sample['y_pred']} "
            f"proba={sample['y_pred_proba']:.4f} positive_trees={sample['positive_voting_tree_count']}"
        )
        if sample.get("y_true") is not None:
            lines.append(f"  y_true={sample['y_true']}")

        for chain in sample["chains"]:
            lines.append(
                f"  - Chain rank {chain['rank']} | tree={chain['tree_index']} "
                f"| leaf_confidence={chain['leaf_confidence']:.4f} | steps={chain['path_length']}"
            )
            for idx, step in enumerate(chain["chain_steps"], start=1):
                lines.append(
                    f"      ({idx}) {step['split']} | MACCS={step['maccs_key']} "
                    f"| {step['substructure_readable']} ({step['substructure_readable_source']}) "
                    f"| {step['interpretation']} "
                    f"| map_conf={step['mapping_confidence']}"
                )
        lines.append("")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TRDP output JSON to model-inferred chemical chain reports."
    )
    parser.add_argument(
        "--trdp-json",
        type=str,
        default="./outputs/trdp_explanations.json",
        help="Path to TRDP explanations JSON.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="./outputs/trdp_summary.json",
        help="Path to TRDP summary JSON for feature-dimension metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for chain reports.",
    )
    parser.add_argument(
        "--maccs-reference-xlsx",
        type=str,
        default="../MACCS_Keys_Human_Readable_Reference(1).xlsx",
        help="Optional MACCS human-readable reference xlsx path.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: str):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    trdp_json = resolve_path(script_dir, args.trdp_json)
    summary_json = resolve_path(script_dir, args.summary_json)
    output_dir = resolve_path(script_dir, args.output_dir)
    maccs_reference_xlsx = resolve_path(script_dir, args.maccs_reference_xlsx)
    output_dir.mkdir(parents=True, exist_ok=True)

    explanations = load_trdp_json(trdp_json)
    summary = load_trdp_json(summary_json)
    total_features = int(summary["n_features"])

    human_reference = load_maccs_human_reference(maccs_reference_xlsx)
    chains = explain_to_chain(
        explanations,
        total_features,
        maccs_human_reference=human_reference,
    )
    report_text = chain_to_text_report(chains, total_features)

    chain_json_path = output_dir / "trdp_chain_report.json"
    chain_txt_path = output_dir / "trdp_chain_report.txt"

    with open(chain_json_path, "w", encoding="utf-8") as handle:
        json.dump(chains, handle, indent=2)

    with open(chain_txt_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    print(f"Saved: {chain_json_path}")
    print(f"Saved: {chain_txt_path}")


if __name__ == "__main__":
    main()
