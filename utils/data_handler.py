from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_csv(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(p)


def ensure_column_exists(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available columns: {list(df.columns)}")
