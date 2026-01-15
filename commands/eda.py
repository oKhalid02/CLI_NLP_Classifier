# commands/eda.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import click
import pandas as pd

from utils.data_handler import load_csv, ensure_column_exists
from utils.visualization import (
    save_pie_chart,
    save_bar_chart,
    save_histogram,
)

# Your topic mapping
TOPIC_MAP = {
    0: "Culture",
    1: "Diverse",
    2: "Economy",
    3: "Politic",
    4: "Sport",
}


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_label_name(x: Any) -> str:
    """
    Convert label value to human-readable name when possible.
    Supports int-like strings.
    """
    try:
        k = int(x)
        return TOPIC_MAP.get(k, str(x))
    except Exception:
        return str(x)


@click.group()
def eda():
    """Exploratory Data Analysis commands"""
    pass


@eda.command()
@click.option("--csv_path", required=True, type=str, help="Path to the CSV file")
@click.option("--text_col", required=True, type=str, help="Name of the text column")
@click.option(
    "--label_col",
    required=False,
    type=str,
    default="target",
    show_default=True,
    help="Name of the label/target column",
)
@click.option(
    "--save_report/--no-save_report",
    default=True,
    show_default=True,
    help="Save a JSON report to outputs/reports/",
)
@click.option(
    "--report_path",
    required=False,
    type=str,
    default="outputs/reports/eda_summary.json",
    show_default=True,
    help="Where to save the EDA JSON report",
)
def summary(
    csv_path: str,
    text_col: str,
    label_col: Optional[str],
    save_report: bool,
    report_path: str,
):
    """
    Basic dataset summary:
    - rows count
    - missing values
    - text length stats (chars + words)
    - label distribution (with names)
    - optionally saves outputs/reports/eda_summary.json
    """
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    click.echo(f"Rows: {len(df)}")
    click.echo(f"Columns: {list(df.columns)}")

    # Missing values
    missing = df.isna().sum().sort_values(ascending=False)
    click.echo("\nMissing values per column:")
    for col, n in missing.items():
        click.echo(f"  - {col}: {n}")

    # Text length stats
    text_series = df[text_col].astype(str).fillna("")
    char_lens = text_series.apply(len)
    word_lens = text_series.apply(lambda s: len(s.split()))

    click.echo("\nText length stats:")
    click.echo(
        f"  - Chars:  mean={char_lens.mean():.2f}, min={char_lens.min()}, max={char_lens.max()}"
    )
    click.echo(
        f"  - Words:  mean={word_lens.mean():.2f}, min={word_lens.min()}, max={word_lens.max()}"
    )

    # Optional label distribution
    label_counts = None
    label_counts_named = None
    if label_col:
        ensure_column_exists(df, label_col)
        label_counts = df[label_col].value_counts(dropna=False).sort_index()
        click.echo("\nLabel distribution:")
        for k, v in label_counts.items():
            click.echo(f"  - {k}: {v} ({_safe_label_name(k)})")

        # Named distribution for report readability
        label_counts_named = {
            f"{k} ({_safe_label_name(k)})": int(v) for k, v in label_counts.items()
        }

    if save_report:
        out_path = Path(report_path)
        _ensure_dir(str(out_path.parent))

        report: Dict[str, Any] = {
            "csv_path": csv_path,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing_values": {k: int(v) for k, v in missing.items()},
            "text_stats": {
                "chars": {
                    "mean": float(char_lens.mean()),
                    "min": int(char_lens.min()),
                    "max": int(char_lens.max()),
                },
                "words": {
                    "mean": float(word_lens.mean()),
                    "min": int(word_lens.min()),
                    "max": int(word_lens.max()),
                },
            },
        }

        if label_col and label_counts is not None:
            report["label_col"] = label_col
            report["label_distribution_raw"] = {
                str(k): int(v) for k, v in label_counts.items()
            }
            report["label_distribution_named"] = label_counts_named or {}

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        click.echo(f"\nReport saved to: {out_path}")


@eda.command()
@click.option("--csv_path", required=True, type=str, help="Path to the CSV file")
@click.option(
    "--label_col",
    required=True,
    type=str,
    help="Name of the label/target column (e.g., target)",
)
@click.option("--plot_type", type=click.Choice(["pie", "bar"]), default="pie", show_default=True)
@click.option("--out_dir", type=str, default="outputs/visualizations", show_default=True)
@click.option("--filename", type=str, default=None, help="Optional custom output filename (png)")
@click.option(
    "--use_topic_names/--no-use_topic_names",
    default=True,
    show_default=True,
    help="Show topic names using TOPIC_MAP in the plot labels",
)
def distribution(
    csv_path: str,
    label_col: str,
    plot_type: str,
    out_dir: str,
    filename: Optional[str],
    use_topic_names: bool,
):
    """
    Plot label distribution as a pie or bar chart.
    Saves to outputs/visualizations by default.
    """
    df = load_csv(csv_path)
    ensure_column_exists(df, label_col)

    counts = df[label_col].astype(str).fillna("NaN").value_counts()

    labels = list(counts.index)
    values = list(counts.values)

    if use_topic_names:
        labels = [f"{x} ({_safe_label_name(x)})" for x in labels]

    if filename is None:
        filename = f"label_distribution_{plot_type}.png"

    if plot_type == "pie":
        out_path = save_pie_chart(
            labels=labels,
            values=values,
            title=f"Label Distribution: {label_col}",
            out_dir=out_dir,
            filename=filename,
        )
    else:
        out_path = save_bar_chart(
            labels=labels,
            values=values,
            title=f"Label Distribution: {label_col}",
            xlabel="Label",
            ylabel="Count",
            out_dir=out_dir,
            filename=filename,
        )

    click.echo(f"Saved: {out_path}")


@eda.command()
@click.option("--csv_path", required=True, type=str, help="Path to the CSV file")
@click.option("--text_col", required=True, type=str, help="Name of the text column")
@click.option("--unit", type=click.Choice(["words", "chars"]), default="words", show_default=True)
@click.option("--bins", type=int, default=30, show_default=True)
@click.option("--out_dir", type=str, default="outputs/visualizations", show_default=True)
@click.option("--filename", type=str, default=None, help="Optional custom output filename (png)")
def histogram(
    csv_path: str,
    text_col: str,
    unit: str,
    bins: int,
    out_dir: str,
    filename: Optional[str],
):
    """
    Plot histogram of text lengths (words or chars).
    Saves to outputs/visualizations by default.
    """
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    text_series = df[text_col].astype(str).fillna("")

    if unit == "words":
        values = text_series.apply(lambda s: len(s.split()))
        title = f"Text Length Histogram (Words): {text_col}"
        xlabel = "Words per row"
        default_name = "text_length_words_hist.png"
    else:
        values = text_series.apply(len)
        title = f"Text Length Histogram (Chars): {text_col}"
        xlabel = "Characters per row"
        default_name = "text_length_chars_hist.png"

    if filename is None:
        filename = default_name

    out_path = save_histogram(
        values=list(values.values),
        bins=bins,
        title=title,
        xlabel=xlabel,
        ylabel="Frequency",
        out_dir=out_dir,
        filename=filename,
    )

    click.echo(f"Saved: {out_path}")
