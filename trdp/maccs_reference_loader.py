from pathlib import Path

import pandas as pd


def _normalize_columns(columns: list[str]) -> dict[str, str]:
    mapping = {}
    for col in columns:
        key = col.strip().lower().replace("-", " ").replace("_", " ")
        key = " ".join(key.split())
        mapping[key] = col
    return mapping


def load_maccs_human_reference(xlsx_path: Path) -> dict[int, dict]:
    if not xlsx_path.exists():
        return {}

    df = pd.read_excel(xlsx_path)
    if df.empty:
        return {}

    normalized = _normalize_columns(list(df.columns))
    bit_col = normalized.get("bit position")
    short_col = normalized.get("short label")
    desc_col = normalized.get("human readable description")

    if bit_col is None:
        return {}

    result = {}
    for _, row in df.iterrows():
        bit_raw = row.get(bit_col)
        if pd.isna(bit_raw):
            continue
        try:
            bit = int(bit_raw)
        except Exception:
            continue

        short_label = ""
        description = ""
        if short_col is not None and not pd.isna(row.get(short_col)):
            short_label = str(row.get(short_col)).strip()
        if desc_col is not None and not pd.isna(row.get(desc_col)):
            description = str(row.get(desc_col)).strip()

        result[bit] = {
            "short_label": short_label,
            "description": description,
        }

    return result
