from __future__ import annotations

import re
import click
import pandas as pd
from pathlib import Path
from typing import Set

from utils.data_handler import load_csv, ensure_column_exists


# -------------------------
# Arabic resources
# -------------------------
ARABIC_STOPWORDS: Set[str] = {
    "من", "إلى", "عن", "على", "في", "هذا", "هذه", "ذلك", "تلك",
    "و", "ثم", "أو", "لكن", "ل", "ب", "ك", "ما", "لا", "لم", "لن",
    "إن", "أن", "كان", "كانت", "هو", "هي", "هم", "هن"
}

TASHKEEL = re.compile(r"[ًٌٍَُِّْ]")
TATWEEL = re.compile(r"ـ")
URLS = re.compile(r"http\S+|www\S+")
DIGITS = re.compile(r"\d+")
SPECIALS = re.compile(r"[^\u0600-\u06FF\s]")


# -------------------------
# Helpers
# -------------------------
def _ensure_out(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _drop_empty(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df[text_col] = df[text_col].astype(str).str.strip()
    return df[df[text_col].str.len() > 0]


def _report(before: pd.Series, after: pd.Series):
    click.echo(f"Rows before: {len(before)}")
    click.echo(f"Rows after : {len(after)}")
    click.echo(
        f"Avg words before: {before.str.split().apply(len).mean():.2f}"
    )
    click.echo(
        f"Avg words after : {after.str.split().apply(len).mean():.2f}"
    )


# -------------------------
# Click group
# -------------------------
@click.group()
def preprocess():
    """Text preprocessing commands"""
    pass


# -------------------------
# remove
# -------------------------
@preprocess.command()
@click.option("--csv_path", required=True, type=str)
@click.option("--text_col", required=True, type=str)
@click.option("--output", required=True, type=str)
def remove(csv_path, text_col, output):
    """Remove tashkeel, tatweel, digits, URLs, special characters"""
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    before = df[text_col].astype(str)

    def clean(text: str) -> str:
        text = TASHKEEL.sub("", text)
        text = TATWEEL.sub("", text)
        text = URLS.sub("", text)
        text = DIGITS.sub("", text)
        text = SPECIALS.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    df[text_col] = before.apply(clean)
    df = _drop_empty(df, text_col)

    after = df[text_col]
    _report(before, after)

    out = _ensure_out(output)
    df.to_csv(out, index=False)
    click.echo(f"Saved → {out}")


# -------------------------
# stopwords
# -------------------------
@preprocess.command()
@click.option("--csv_path", required=True, type=str)
@click.option("--text_col", required=True, type=str)
@click.option("--output", required=True, type=str)
def stopwords(csv_path, text_col, output):
    """Remove Arabic stopwords"""
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    before = df[text_col].astype(str)

    def remove_sw(text: str) -> str:
        tokens = [w for w in text.split() if w not in ARABIC_STOPWORDS]
        return " ".join(tokens)

    df[text_col] = before.apply(remove_sw)
    df = _drop_empty(df, text_col)

    after = df[text_col]
    _report(before, after)

    out = _ensure_out(output)
    df.to_csv(out, index=False)
    click.echo(f"Saved → {out}")


# -------------------------
# replace
# -------------------------
@preprocess.command()
@click.option("--csv_path", required=True, type=str)
@click.option("--text_col", required=True, type=str)
@click.option("--output", required=True, type=str)
def replace(csv_path, text_col, output):
    """Normalize Arabic characters"""
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    before = df[text_col].astype(str)

    def normalize(text: str) -> str:
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("ؤ", "و", text)
        text = re.sub("ئ", "ي", text)
        return text

    df[text_col] = before.apply(normalize)
    df = _drop_empty(df, text_col)

    after = df[text_col]
    _report(before, after)

    out = _ensure_out(output)
    df.to_csv(out, index=False)
    click.echo(f"Saved → {out}")


# -------------------------
# all (chain)
# -------------------------
@preprocess.command()
@click.option("--csv_path", required=True, type=str)
@click.option("--text_col", required=True, type=str)
@click.option("--output", required=True, type=str)
def all(csv_path, text_col, output):
    """Run remove → stopwords → replace"""
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    before = df[text_col].astype(str)

    def pipeline(text: str) -> str:
        text = TASHKEEL.sub("", text)
        text = TATWEEL.sub("", text)
        text = URLS.sub("", text)
        text = DIGITS.sub("", text)
        text = SPECIALS.sub(" ", text)

        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("ؤ", "و", text)
        text = re.sub("ئ", "ي", text)

        tokens = [w for w in text.split() if w not in ARABIC_STOPWORDS]
        return " ".join(tokens)

    df[text_col] = before.apply(pipeline)
    df = _drop_empty(df, text_col)

    after = df[text_col]
    _report(before, after)

    out = _ensure_out(output)
    df.to_csv(out, index=False)
    click.echo(f"Saved → {out}")
