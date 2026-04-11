# DDI Term Project

Drug–drug interaction (DDI) prediction from molecular fingerprints (MACCS keys), with classical and neural baselines, SHAP explainability, TRDP path extraction, and optional SHAP-guided Random Forest experiments.

---

## Table of contents

1. [What this repository contains](#what-this-repository-contains)
2. [Prerequisites](#prerequisites)
3. [Clone and Git LFS](#clone-and-git-lfs)
4. [Python environment](#python-environment)
5. [Install dependencies](#install-dependencies)
6. [Project layout](#project-layout)
7. [Data you need at the repository root](#data-you-need-at-the-repository-root)
8. [Reproducing processed datasets](#reproducing-processed-datasets)
9. [Training and evaluation](#training-and-evaluation)
10. [Explainability](#explainability)
11. [Guided TRDP experiment (two-stage RF)](#guided-trdp-experiment-two-stage-rf)
12. [Troubleshooting](#troubleshooting)

---

## What this repository contains

**Version-controlled**

- Source code under `pipelines/`, `trdp/`, and `project_paths.py`.
- **Raw inputs**: `ChCh-Miner_durgbank-chem-chem.tsv/`, `drug_smiles.json`.
- **Processed tensors** under `data/processed/` (`*.npz`, `*.npy`) stored with **Git LFS** (see `.gitattributes`).
- **MACCS human-readable reference** (optional but recommended for reports): `MACCS_Keys_Human_Readable_Reference(1).xlsx`.
- Progress notes: `REPORT_guided_trdp_progress.md`, `PLAN_guided_trdp.md`, `PROJECT_STRUCTURE.md`.

**Not version-controlled (regenerate locally)**

- `artifacts/` (metrics JSON, figures, SHAP exports, guided-TRDP CSV/JSON, etc.).
- `trdp/outputs/` if you override TRDP scripts to write there.
- Model checkpoints: `*.pkl`, `*.pth` (see `.gitignore`).

Teammates should **clone, pull LFS, install dependencies, then run scripts** to recreate metrics and models.

---

## Prerequisites

- **Python** 3.10 or newer (3.11 recommended).
- **Git** with **Git LFS** installed (`git lfs install` once per machine).
- **RDKit** (via Conda is common; the `pip` wheel below also works on many platforms).
- Optional: **CUDA**-capable GPU for PyTorch models (`pipelines/models/*` deep learning scripts).

---

## Clone and Git LFS

```bash
git clone https://github.com/SAKICHANNN/DDI-term-project.git
cd DDI-term-project
git lfs install
git lfs pull
```

If `data/processed/*.npy` / `*.npz` are missing or tiny text stubs after clone, LFS was not pulled; run `git lfs pull` again.

---

## Python environment

Create and activate a virtual environment (recommended):

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install dependencies

There is no pinned `requirements.txt` in this repository; install the following (versions are indicative):

```bash
pip install "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3" "torch>=2.1" "shap>=0.44" "openpyxl>=3.1"
```

**RDKit** (required for `pipelines/data/encode.py`):

```bash
pip install rdkit-pypi
```

If `rdkit-pypi` fails on your platform, install RDKit from [Conda Forge](https://www.rdkit.org/docs/Install.html) and use that environment instead.

---

## Project layout

| Path | Role |
|------|------|
| `project_paths.py` | Single source of truth for input/output paths. |
| `data/processed/` | Fingerprints, pair samples, train/val/test arrays. |
| `artifacts/models/` | Saved `*.pkl` / `*.pth` after you train (not in Git). |
| `artifacts/metrics/` | JSON metrics after training. |
| `artifacts/figures/` | SHAP plots, etc. |
| `artifacts/explanations/` | SHAP- or TRDP-related exports, guided-TRDP runs. |
| `pipelines/data/` | Dataset construction scripts. |
| `pipelines/models/` | LR, RF, MLP, attention variants. |
| `pipelines/explainability/` | SHAP analysis. |
| `pipelines/experiments/guided_trdp/` | Guided RF + evaluation driver. |
| `trdp/` | TRDP analysis, chain reports, mechanism helper scripts. |

More detail: `PROJECT_STRUCTURE.md`.

---

## Data you need at the repository root

| File / directory | Purpose |
|------------------|---------|
| `drug_smiles.json` | DrugBank ID → SMILES mapping for fingerprint encoding. |
| `ChCh-Miner_durgbank-chem-chem.tsv/ChCh-Miner_durgbank-chem-chem.tsv` | Positive DDI pairs (TSV). |
| `MACCS_Keys_Human_Readable_Reference(1).xlsx` | Optional: human-readable MACCS labels for TRDP reports. |

If `data/processed/` is already populated from Git LFS, you can **skip** the full rebuild in the next section and go straight to training—unless you want to regenerate everything from raw files.

---

## Reproducing processed datasets

Always run scripts from the **repository root** so `project_paths.py` resolves correctly.

### A) Concatenated fingerprints (334 → 324 features after variance filter)

This pipeline **overwrites** `data/processed/X_*.npy` and `y_*.npy`.

```bash
python pipelines/data/encode.py
python pipelines/data/positive_construct.py
python pipelines/data/negative_construct.py
python pipelines/data/dataset_create.py
```

### B) Additive fingerprints (167 features)

Run this **after** you no longer need the concat matrices for the same session, or copy outputs aside first—it also **overwrites** the same `X_*.npy` / `y_*.npy` filenames.

```bash
python pipelines/data/encode.py
python pipelines/data/positive_add.py
python pipelines/data/negative_add.py
python pipelines/data/dataset_new_create.py
```

Intermediate NPZ files:

- Concat: `positive_samples.npz`, `negative_samples.npz`.
- Add: `positive_add_samples.npz`, `negative_add_samples.npz`.

---

## Training and evaluation

Activate your virtual environment, then from the repo root:

```bash
python pipelines/models/logistic_regression.py
python pipelines/models/random_forest.py
python pipelines/models/mlp.py
python pipelines/models/attention.py
python pipelines/models/full_attention.py
python pipelines/models/bi_full_attention.py
```

Outputs go to `artifacts/models/` and `artifacts/metrics/` (and figures where implemented). Scripts import `project_paths` for destinations.

---

## Explainability

### SHAP (Random Forest)

```bash
python pipelines/explainability/shap_analysis.py
```

Uses the trained RF and processed splits; figures typically go to `artifacts/figures/`.

### TRDP (Random Forest paths)

Defaults (see `trdp/trdp_analysis.py`): model `artifacts/models/rf_model.pkl`, features `data/processed/X_test.npy`, labels `data/processed/y_test.npy`, output directory `artifacts/explanations/trdp/`.

```bash
python trdp/trdp_analysis.py --only-positive-predictions --top-k 3 --max-samples 20
```

Then enrich paths with MACCS text (optional Excel):

```bash
python trdp/trdp_chain_report.py --maccs-reference-xlsx "MACCS_Keys_Human_Readable_Reference(1).xlsx"
```

Other helpers (see `trdp/README.md` and `--help` on each file):

- `trdp/trdp_conclusion.py`
- `trdp/trdp_pair_conclusion.py`
- `trdp/mechanism_hypothesis.py`

---

## Guided TRDP experiment (two-stage RF)

Requires the **same** `data/processed/X_*.npy` split you want to study (concat or add pipeline). Outputs go to `artifacts/explanations/<run-name>/`.

Example (quick run; use one line, or use your shell’s line-continuation character—`^` in **cmd.exe**, `` ` `` in **PowerShell**):

```bash
python pipelines/experiments/guided_trdp/run_guided_trdp_experiment.py --seeds 42 --alphas 0,0.7 --top-k 1 --top-n 10 --shap-max-samples 150 --run-name my_guided_run
```

Full CLI options:

```bash
python pipelines/experiments/guided_trdp/run_guided_trdp_experiment.py --help
```

Design notes and metric definitions: `REPORT_guided_trdp_progress.md`, `PLAN_guided_trdp.md`.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| LFS files missing / corrupt pointers | `git lfs install && git lfs pull`; ensure Git LFS is installed before clone. |
| `ModuleNotFoundError: rdkit` | Install RDKit (`rdkit-pypi` or Conda). |
| `FileNotFoundError` for `X_train.npy` | Run the appropriate dataset pipeline (section [Reproducing processed datasets](#reproducing-processed-datasets)). |
| Wrong feature dimension for TRDP / RF | You switched concat ↔ add without rebuilding; rerun the correct `dataset_*.py` script. |
| TRDP paths look wrong | Ensure `rf_model.pkl` matches the same feature layout as `X_test.npy` (same pipeline). |
| PyTorch CUDA errors | Install a CUDA build of PyTorch matching your driver, or run CPU-only (slower). |
| Git push rejected for large files | Do not commit `*.pkl` / large binaries; keep them under `artifacts/models/` locally. Large arrays belong in LFS (`.gitattributes`). |

---

## License and attribution

Course project repository. Cite original data sources (DrugBank-derived pairs file, SMILES mapping) and this codebase as required by your institution.
