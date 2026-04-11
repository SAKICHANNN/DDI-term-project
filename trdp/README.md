# TRDP Module

This folder contains a standalone implementation of TRDP (Tree-Ranked Decision Path Output) for the Random Forest model in this project.

## What it does

`trdp_analysis.py` generates per-sample explanations by:

1. collecting all trees that vote positive (`leaf_confidence > 0.5`),
2. ranking those trees by leaf confidence,
3. outputting the complete root-to-leaf path for top-K trees.

Each path is saved as an ordered list of conditions in a JSON file.

## Prerequisites

- A trained Random Forest model: `rf_model.pkl`
- Dataset arrays, usually:
  - `X_test.npy`
  - `y_test.npy` (optional but recommended)

## Usage

From project root:

```bash
python trdp/trdp_analysis.py --only-positive-predictions --top-k 3 --max-samples 20
```

You can also target specific samples:

```bash
python trdp/trdp_analysis.py --sample-indices 1,5,42 --top-k 5
```

## Main arguments

- `--model-path`: path to Random Forest pickle (default `../rf_model.pkl` relative to this script)
- `--x-path`: path to feature matrix (default `../X_test.npy`)
- `--y-path`: path to labels (default `../y_test.npy`)
- `--output-dir`: output folder (default `./outputs` inside this folder)
- `--top-k`: number of ranked trees per sample
- `--max-samples`: max sample count when auto-selecting
- `--only-positive-predictions`: explain only samples predicted as positive
- `--sample-indices`: comma-separated list of specific indices

## Outputs

By default, outputs are written to `trdp/outputs/`:

- `trdp_explanations.json`: detailed TRDP paths per explained sample
- `trdp_summary.json`: run configuration and summary stats

## Generate chemical-chain style report

After `trdp_explanations.json` is generated, run:

```bash
python trdp/trdp_chain_report.py
```

If you have a custom human-readable MACCS table:

```bash
python trdp/trdp_chain_report.py --maccs-reference-xlsx "../MACCS_Keys_Human_Readable_Reference(1).xlsx"
```

This creates:

- `trdp_chain_report.json`: TRDP paths enriched with MACCS/SMARTS metadata
- `trdp_chain_report.txt`: readable rule-chain report for each sample

## Generate one-line conflict conclusion

You can generate a compact conclusion in the style:

`Drug A and Drug B are predicted to conflict because ...`

Example:

```bash
python trdp/trdp_conclusion.py --sample-index 6 --drug-a-name "Drug A" --drug-b-name "Drug B"
```

Output:

- `trdp_conclusion.txt`

## Generate conclusion from DrugBank IDs

You can directly ask for a pair-level conclusion by DrugBank IDs:

```bash
python trdp/trdp_pair_conclusion.py --drug-a-id DB00862 --drug-b-id DB00966 --top-k 3
```

To force your own MACCS readable table:

```bash
python trdp/trdp_pair_conclusion.py --drug-a-id DB00862 --drug-b-id DB00966 --maccs-reference-xlsx "../MACCS_Keys_Human_Readable_Reference(1).xlsx"
```

Outputs:

- `pair_conclusion_<DrugA>_<DrugB>.json`
- `pair_conclusion_<DrugA>_<DrugB>.txt`

The text output uses a more human-readable substructure phrasing rather than raw SMARTS only.

## Advanced chemistry-oriented hypothesis (all six extensions)

This command generates a single report with:

1. structural co-occurrence motifs,
2. mechanism label hypotheses,
3. counterfactual edits,
4. TRDP + SHAP consistency (plus optional attention consistency),
5. semantic layer summary (SMILES + RDKit descriptors + BIOSNAP pair presence),
6. uncertainty/confidence grading.

```bash
python trdp/mechanism_hypothesis.py --drug-a-id DB00862 --drug-b-id DB00966 --top-k 3 --maccs-reference-xlsx "../MACCS_Keys_Human_Readable_Reference(1).xlsx"
```

Outputs:

- `mechanism_hypothesis_<DrugA>_<DrugB>.json`
- `mechanism_hypothesis_<DrugA>_<DrugB>.txt`

Optional:

- Put `attention_feature_importance.json` under `trdp/knowledge/` to enable third-source (attention) overlap checks.
- See `trdp/knowledge/attention_feature_importance.example.json` for format.

Important:

- This report is a **model-inferred rule chain**, not a proven biochemical mechanism.
- For reduced feature spaces (e.g., after variance filtering), MACCS key mapping is marked with lower confidence.

## Notes

- Feature names are generated from column positions.
- If the feature dimension is even, names are split as `DrugA_F*` and `DrugB_F*`.
- If the dimension is odd (e.g., additive representation), names are `Feature_*`.
