import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pick_key_steps(chain_steps: list[dict], max_points: int = 5) -> list[dict]:
    contains = []
    excludes = []
    for step in chain_steps:
        text = step.get("interpretation", "")
        if "contains this substructure" in text:
            contains.append(step)
        elif "does not contain this substructure" in text:
            excludes.append(step)

    selected = contains[: max_points - 2] if len(contains) >= max_points - 2 else contains
    remaining = max_points - len(selected)
    selected += excludes[:remaining]
    return selected


def format_step(step: dict) -> str:
    side = "Drug A" if step.get("feature_name", "").startswith("DrugA_") else (
        "Drug B" if step.get("feature_name", "").startswith("DrugB_") else "Pair"
    )
    maccs = step.get("maccs_key")
    readable = step.get("substructure_readable", "an unresolved substructure pattern")
    readable_phrase = readable.strip()
    if readable_phrase.lower().startswith("contains "):
        readable_phrase = readable_phrase[9:]
    if readable_phrase:
        readable_phrase = readable_phrase[0].lower() + readable_phrase[1:]

    base = step.get("interpretation", "")
    if "contains this substructure" in base:
        interpretation = f"{side} contains {readable_phrase}"
    elif "does not contain this substructure" in base:
        interpretation = f"{side} does not contain {readable_phrase}"
    else:
        interpretation = base.replace("this substructure", readable_phrase)
    return f"{side} [{step.get('feature_name')}] (MACCS {maccs}): {interpretation}"


def generate_conclusion(sample: dict, drug_a_name: str, drug_b_name: str) -> str:
    if not sample.get("chains"):
        return (
            f"{drug_a_name} and {drug_b_name} are not supported by a positive TRDP chain "
            f"in the selected sample."
        )

    top_chain = sample["chains"][0]
    key_steps = pick_key_steps(top_chain.get("chain_steps", []), max_points=5)

    header = (
        f"{drug_a_name} and {drug_b_name} are predicted to conflict "
        f"(probability={sample['y_pred_proba']:.3f}, positive_voting_trees={sample['positive_voting_tree_count']}) "
        f"because the top-ranked decision path (tree={top_chain['tree_index']}, "
        f"leaf_confidence={top_chain['leaf_confidence']:.3f}) captures the following structural pattern:"
    )
    bullets = "\n".join([f"- {format_step(step)}" for step in key_steps])
    tail = (
        "\nThis is a model-inferred interaction hypothesis based on fingerprint rules, "
        "not a laboratory-confirmed biochemical mechanism."
    )
    return f"{header}\n{bullets}{tail}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate concise human-readable DDI conclusion from TRDP chain output."
    )
    parser.add_argument(
        "--chain-json",
        type=str,
        default="./outputs/trdp_chain_report.json",
        help="Path to trdp_chain_report.json",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index in the chain report",
    )
    parser.add_argument(
        "--drug-a-name",
        type=str,
        default="Drug A",
        help="Display name for drug A",
    )
    parser.add_argument(
        "--drug-b-name",
        type=str,
        default="Drug B",
        help="Display name for drug B",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/trdp_conclusion.txt",
        help="Output text file path",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: str):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    chain_json = resolve_path(base_dir, args.chain_json)
    output_path = resolve_path(base_dir, args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_json(chain_json)
    target = None
    for sample in samples:
        if int(sample["sample_index"]) == int(args.sample_index):
            target = sample
            break

    if target is None:
        raise ValueError(f"Sample index {args.sample_index} not found in {chain_json}")

    conclusion = generate_conclusion(
        sample=target,
        drug_a_name=args.drug_a_name,
        drug_b_name=args.drug_b_name,
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(conclusion + "\n")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
