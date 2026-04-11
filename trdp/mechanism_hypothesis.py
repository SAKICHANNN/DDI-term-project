import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.feature_selection import VarianceThreshold

from maccs_reference_loader import load_maccs_human_reference
from trdp_analysis import build_feature_names, explain_sample
from trdp_chain_report import condition_to_text, load_maccs_patterns


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


def build_pair_feature(
    rf_model,
    drug_a_id: str,
    drug_b_id: str,
    fingerprints_path: Path,
    positive_samples_path: Path,
    negative_samples_path: Path,
):
    fp_data = np.load(fingerprints_path, allow_pickle=True)
    fingerprints = dict(zip(fp_data["ids"], fp_data["fps"]))

    if drug_a_id not in fingerprints:
        raise ValueError(f"Drug A id not found in fingerprints: {drug_a_id}")
    if drug_b_id not in fingerprints:
        raise ValueError(f"Drug B id not found in fingerprints: {drug_b_id}")

    raw_feature = np.concatenate([fingerprints[drug_a_id], fingerprints[drug_b_id]]).astype(float)
    model_dim = int(rf_model.n_features_in_)

    if raw_feature.shape[0] == model_dim:
        return raw_feature, "direct"

    mask = load_selector_mask(positive_samples_path, negative_samples_path)
    reduced_feature = raw_feature[mask]
    if reduced_feature.shape[0] != model_dim:
        raise ValueError(
            f"Feature mismatch after reduction: model={model_dim}, reduced={reduced_feature.shape[0]}"
        )
    return reduced_feature, "variance_threshold_mask"


def enrich_chains(explanation: dict, model_dim: int, human_reference: dict[int, dict]):
    patterns = load_maccs_patterns()
    enriched = []
    for chain in explanation["ranked_paths"]:
        steps = [
            condition_to_text(
                condition,
                model_dim,
                patterns,
                maccs_human_reference=human_reference,
            )
            for condition in chain["conditions"]
        ]
        enriched.append(
            {
                "rank": chain["rank"],
                "tree_index": chain["tree_index"],
                "leaf_confidence": chain["leaf_confidence"],
                "path_length": chain["path_length"],
                "chain_steps": steps,
            }
        )
    return enriched


def mine_co_occurrence_motifs(chains: list[dict], top_n: int = 10):
    motif_counter = {}
    for chain in chains:
        weight = float(chain["leaf_confidence"])
        contains = []
        for step in chain["chain_steps"]:
            if "contains this substructure" in step["interpretation"]:
                token = f"{step['feature_name']}::{step['substructure_readable']}"
                contains.append(token)

        for token in contains:
            motif_counter[token] = motif_counter.get(token, 0.0) + weight

        for a, b in combinations(sorted(set(contains)), 2):
            key = f"{a} && {b}"
            motif_counter[key] = motif_counter.get(key, 0.0) + weight

    ranked = sorted(motif_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"motif": k, "score": round(v, 4)} for k, v in ranked]


def infer_mechanism_labels(chains: list[dict]):
    text = " ".join(
        step["substructure_readable"].lower()
        for chain in chains
        for step in chain["chain_steps"]
    )
    labels = []

    def add(label: str, why: str, confidence: str):
        labels.append({"label": label, "reason": why, "confidence": confidence})

    has_oxygen = "oxygen" in text
    has_nitrogen = "nitrogen" in text or "amine" in text
    has_sulfur = "sulfur" in text
    has_halogen = "halogen" in text or "fluor" in text or "chlor" in text or "brom" in text
    has_ring = "ring" in text or "aromatic" in text

    if has_oxygen and has_nitrogen:
        add(
            "Metabolic competition proxy (CYP-related hypothesis)",
            "Recurring oxygen and nitrogen motifs often track oxidative metabolism-related risk patterns.",
            "medium",
        )
    if has_ring and has_halogen:
        add(
            "Transporter/binding perturbation proxy",
            "Ring-rich and halogenated motifs frequently indicate strong binding and distribution effects.",
            "low",
        )
    if has_sulfur and has_oxygen:
        add(
            "Sulfur-oxygen functional interplay hypothesis",
            "Sulfur and oxygen pattern co-occurrence appears repeatedly in top decision chains.",
            "low",
        )

    if not labels:
        add(
            "General structural incompatibility hypothesis",
            "No strong mechanism signature found, but multi-tree structural evidence supports interaction.",
            "low",
        )

    return labels


def run_counterfactual(rf_model, sample: np.ndarray, chains: list[dict], top_n: int = 8):
    base = float(rf_model.predict_proba(sample.reshape(1, -1))[0][1])
    candidate_indices = set()
    for chain in chains:
        for step in chain["chain_steps"]:
            candidate_indices.add(int(step["feature_name"].split("_F")[-1]))

    impacts = []
    for idx in sorted(candidate_indices):
        if idx < 0 or idx >= sample.shape[0]:
            continue
        edited = sample.copy()
        edited[idx] = 1.0 - edited[idx]
        new_prob = float(rf_model.predict_proba(edited.reshape(1, -1))[0][1])
        impacts.append(
            {
                "feature_index": int(idx),
                "old_value": float(sample[idx]),
                "new_value": float(edited[idx]),
                "new_probability": round(new_prob, 6),
                "delta_probability": round(new_prob - base, 6),
            }
        )

    impacts.sort(key=lambda x: x["delta_probability"])
    return {
        "base_probability": round(base, 6),
        "top_risk_reducing_edits": impacts[:top_n],
        "top_risk_increasing_edits": sorted(impacts, key=lambda x: x["delta_probability"], reverse=True)[:top_n],
    }


def compute_evidence_consistency(rf_model, sample: np.ndarray, chains: list[dict], attention_hint_path: Path):
    trdp_features = set()
    for chain in chains:
        for step in chain["chain_steps"]:
            trdp_features.add(step["feature_name"])

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(sample.reshape(1, -1))
    if isinstance(shap_values, list):
        shap_pos = shap_values[1][0]
    else:
        shap_pos = shap_values[0, :, 1]

    top_idx = np.argsort(np.abs(shap_pos))[::-1][:20]
    feature_names = build_feature_names(sample.shape[0])
    shap_top_features = [feature_names[int(i)] for i in top_idx]
    shap_set = set(shap_top_features)

    intersection = trdp_features & shap_set
    union = trdp_features | shap_set
    jaccard = float(len(intersection) / len(union)) if union else 0.0

    attention_consistency = {
        "status": "unavailable",
        "note": "No attention feature attribution file found; skipped third-source consistency.",
    }
    if attention_hint_path.exists():
        try:
            hint = json.load(open(attention_hint_path, "r", encoding="utf-8"))
            att_features = set(hint.get("top_features", []))
            if att_features:
                inter_att = trdp_features & att_features
                attention_consistency = {
                    "status": "available",
                    "overlap_with_trdp": sorted(inter_att),
                    "overlap_count": len(inter_att),
                }
        except Exception:
            pass

    return {
        "trdp_feature_count": len(trdp_features),
        "shap_feature_count": len(shap_set),
        "trdp_shap_overlap": sorted(intersection),
        "trdp_shap_overlap_count": len(intersection),
        "trdp_shap_jaccard": round(jaccard, 4),
        "attention_consistency": attention_consistency,
    }


def rdkit_semantic_summary(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid_smiles": False}
    return {
        "valid_smiles": True,
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 3),
        "logp": round(float(Descriptors.MolLogP(mol)), 3),
        "h_bond_donor": int(Descriptors.NumHDonors(mol)),
        "h_bond_acceptor": int(Descriptors.NumHAcceptors(mol)),
        "ring_count": int(Descriptors.RingCount(mol)),
        "aromatic_ring_count": int(Descriptors.NumAromaticRings(mol)),
    }


def build_semantic_layer(drug_a_id: str, drug_b_id: str, drug_smiles_path: Path, biosnap_path: Path):
    smiles_map = json.load(open(drug_smiles_path, "r", encoding="utf-8"))
    smiles_a = smiles_map.get(drug_a_id)
    smiles_b = smiles_map.get(drug_b_id)

    in_biosnap = False
    if biosnap_path.exists():
        df = pd.read_csv(biosnap_path, sep="\t", header=None, names=["drug1", "drug2"])
        in_biosnap = bool(
            ((df["drug1"] == drug_a_id) & (df["drug2"] == drug_b_id)).any()
            or ((df["drug1"] == drug_b_id) & (df["drug2"] == drug_a_id)).any()
        )

    return {
        "drug_a_id": drug_a_id,
        "drug_b_id": drug_b_id,
        "known_positive_pair_in_biosnap": in_biosnap,
        "drug_a_smiles": smiles_a,
        "drug_b_smiles": smiles_b,
        "drug_a_rdkit_summary": rdkit_semantic_summary(smiles_a) if smiles_a else {"valid_smiles": False},
        "drug_b_rdkit_summary": rdkit_semantic_summary(smiles_b) if smiles_b else {"valid_smiles": False},
    }


def grade_uncertainty(prob: float, vote_ratio: float, jaccard: float, counterfactual: dict):
    best_drop = 0.0
    if counterfactual["top_risk_reducing_edits"]:
        best_drop = -float(counterfactual["top_risk_reducing_edits"][0]["delta_probability"])

    score = 0.0
    score += min(max((prob - 0.5) * 2, 0.0), 1.0) * 0.4
    score += min(max(vote_ratio, 0.0), 1.0) * 0.25
    score += min(max(jaccard * 2, 0.0), 1.0) * 0.2
    score += min(max(best_drop, 0.0), 0.5) * 0.15

    if score >= 0.7:
        grade = "high_confidence_hypothesis"
    elif score >= 0.45:
        grade = "medium_confidence_hypothesis"
    else:
        grade = "low_confidence_hypothesis"

    return {
        "confidence_grade": grade,
        "confidence_score": round(score, 4),
        "inputs": {
            "probability": round(prob, 4),
            "vote_ratio": round(vote_ratio, 4),
            "trdp_shap_jaccard": round(jaccard, 4),
            "best_counterfactual_drop": round(best_drop, 4),
        },
    }


def write_text_report(path: Path, report: dict):
    lines = []
    p = report["prediction"]
    lines.append(
        f"{p['drug_a_id']} and {p['drug_b_id']} -> predicted {'conflict' if p['y_pred'] == 1 else 'no conflict'} "
        f"(probability={p['y_pred_proba']:.3f}, positive_voting_trees={p['positive_voting_tree_count']})."
    )
    lines.append("")
    lines.append("1) Structural co-occurrence motifs:")
    for item in report["co_occurrence_motifs"][:8]:
        lines.append(f"- score={item['score']}: {item['motif']}")

    lines.append("")
    lines.append("2) Mechanism labels (hypothesis):")
    for item in report["mechanism_labels"]:
        lines.append(f"- {item['label']} ({item['confidence']}): {item['reason']}")

    lines.append("")
    lines.append("3) Counterfactual edits that reduce predicted risk:")
    for item in report["counterfactual"]["top_risk_reducing_edits"][:6]:
        lines.append(
            f"- flip F{item['feature_index']} ({item['old_value']}->{item['new_value']}): "
            f"delta={item['delta_probability']}"
        )

    lines.append("")
    lines.append("4) Evidence consistency:")
    c = report["evidence_consistency"]
    lines.append(
        f"- TRDP/SHAP overlap: {c['trdp_shap_overlap_count']} features, jaccard={c['trdp_shap_jaccard']}"
    )
    lines.append(f"- Attention consistency: {c['attention_consistency']['status']}")
    lines.append(f"  note: {c['attention_consistency']['note']}")

    lines.append("")
    lines.append("5) Drug semantic layer:")
    s = report["semantic_layer"]
    lines.append(f"- Known positive in BIOSNAP: {s['known_positive_pair_in_biosnap']}")
    lines.append(f"- Drug A summary: {s['drug_a_rdkit_summary']}")
    lines.append(f"- Drug B summary: {s['drug_b_rdkit_summary']}")

    lines.append("")
    u = report["uncertainty_grade"]
    lines.append("6) Uncertainty grade:")
    lines.append(f"- {u['confidence_grade']} (score={u['confidence_score']})")
    lines.append(f"- factors: {u['inputs']}")

    lines.append("")
    lines.append("Note: This report is for model-based hypothesis generation, not confirmed biochemical causality.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate advanced chemistry-oriented mechanism hypothesis from TRDP outputs."
    )
    parser.add_argument("--drug-a-id", type=str, required=True)
    parser.add_argument("--drug-b-id", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-path", type=str, default="../rf_model.pkl")
    parser.add_argument("--fingerprints-path", type=str, default="../drug_fingerprints.npz")
    parser.add_argument("--positive-samples-path", type=str, default="../positive_samples.npz")
    parser.add_argument("--negative-samples-path", type=str, default="../negative_samples.npz")
    parser.add_argument("--drug-smiles-path", type=str, default="../drug_smiles.json")
    parser.add_argument(
        "--biosnap-path",
        type=str,
        default="../ChCh-Miner_durgbank-chem-chem.tsv/ChCh-Miner_durgbank-chem-chem.tsv",
    )
    parser.add_argument(
        "--maccs-reference-xlsx",
        type=str,
        default="../MACCS_Keys_Human_Readable_Reference(1).xlsx",
    )
    parser.add_argument(
        "--attention-hint-path",
        type=str,
        default="./knowledge/attention_feature_importance.json",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    model_path = resolve_path(base_dir, args.model_path)
    fingerprints_path = resolve_path(base_dir, args.fingerprints_path)
    pos_path = resolve_path(base_dir, args.positive_samples_path)
    neg_path = resolve_path(base_dir, args.negative_samples_path)
    smiles_path = resolve_path(base_dir, args.drug_smiles_path)
    biosnap_path = resolve_path(base_dir, args.biosnap_path)
    xlsx_path = resolve_path(base_dir, args.maccs_reference_xlsx)
    attention_hint_path = resolve_path(base_dir, args.attention_hint_path)
    output_dir = resolve_path(base_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as handle:
        rf_model = pickle.load(handle)

    feature, transform_mode = build_pair_feature(
        rf_model,
        args.drug_a_id,
        args.drug_b_id,
        fingerprints_path,
        pos_path,
        neg_path,
    )

    feature_names = build_feature_names(feature.shape[0])
    explanation = explain_sample(
        rf_model=rf_model,
        sample=feature,
        sample_index=-1,
        feature_names=feature_names,
        top_k=args.top_k,
        y_true=None,
    )

    human_reference = load_maccs_human_reference(xlsx_path)
    chains = enrich_chains(explanation, feature.shape[0], human_reference)
    motifs = mine_co_occurrence_motifs(chains, top_n=12)
    labels = infer_mechanism_labels(chains)
    counterfactual = run_counterfactual(rf_model, feature, chains, top_n=8)
    consistency = compute_evidence_consistency(rf_model, feature, chains, attention_hint_path)
    semantic_layer = build_semantic_layer(args.drug_a_id, args.drug_b_id, smiles_path, biosnap_path)

    vote_ratio = float(explanation["positive_voting_tree_count"] / len(rf_model.estimators_))
    uncertainty = grade_uncertainty(
        prob=float(explanation["y_pred_proba"]),
        vote_ratio=vote_ratio,
        jaccard=float(consistency["trdp_shap_jaccard"]),
        counterfactual=counterfactual,
    )

    report = {
        "prediction": {
            "drug_a_id": args.drug_a_id,
            "drug_b_id": args.drug_b_id,
            "y_pred": explanation["y_pred"],
            "y_pred_proba": explanation["y_pred_proba"],
            "positive_voting_tree_count": explanation["positive_voting_tree_count"],
            "top_k": args.top_k,
            "feature_transform_mode": transform_mode,
        },
        "chains": chains,
        "co_occurrence_motifs": motifs,
        "mechanism_labels": labels,
        "counterfactual": counterfactual,
        "evidence_consistency": consistency,
        "semantic_layer": semantic_layer,
        "uncertainty_grade": uncertainty,
    }

    stem = f"{args.drug_a_id}_{args.drug_b_id}"
    json_path = output_dir / f"mechanism_hypothesis_{stem}.json"
    txt_path = output_dir / f"mechanism_hypothesis_{stem}.txt"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_text_report(txt_path, report)

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
