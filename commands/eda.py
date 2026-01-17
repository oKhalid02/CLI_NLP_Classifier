# commands/eda.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any

import click
import pandas as pd
import numpy as np

from utils.data_handler import load_csv, ensure_column_exists
from utils.visualization import (
    save_pie_chart,
    save_bar_chart,
    save_histogram,
)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _series_to_str(s: pd.Series) -> pd.Series:
    # Keep labels "as-is" but make them printable (also handles ints, floats, etc.)
    return s.astype(str).fillna("NaN")


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
    default=None,
    help="(Optional) Name of the label column. If not provided, label distribution is skipped.",
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
    - (optional) label distribution (DYNAMIC: raw label values)
    - saves outputs/reports/eda_summary.json by default
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
    click.echo(f"  - Chars:  mean={char_lens.mean():.2f}, min={char_lens.min()}, max={char_lens.max()}")
    click.echo(f"  - Words:  mean={word_lens.mean():.2f}, min={word_lens.min()}, max={word_lens.max()}")

    # Optional label distribution (DYNAMIC)
    label_counts = None
    if label_col:
        ensure_column_exists(df, label_col)
        label_str = _series_to_str(df[label_col])
        label_counts = label_str.value_counts(dropna=False)

        click.echo("\nLabel distribution (raw labels):")
        for k, v in label_counts.items():
            click.echo(f"  - {k}: {v}")

    if save_report:
        out_path = Path(report_path)
        _ensure_dir(str(out_path.parent))

        report = {
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
            report["label_distribution"] = {str(k): int(v) for k, v in label_counts.items()}

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        click.echo(f"\nReport saved to: {out_path}")


@eda.command()
@click.option("--csv_path", required=True, type=str, help="Path to the CSV file")
@click.option("--label_col", required=True, type=str, help="Name of the label column")
@click.option("--plot_type", type=click.Choice(["pie", "bar"]), default="pie", show_default=True)
@click.option("--out_dir", type=str, default="outputs/visualizations", show_default=True)
@click.option("--filename", type=str, default=None, help="Optional custom output filename (png)")
def distribution(
    csv_path: str,
    label_col: str,
    plot_type: str,
    out_dir: str,
    filename: Optional[str],
):
    """
    Plot label distribution as a pie or bar chart.
    DYNAMIC: uses raw labels exactly as they appear in the dataset.
    """
    df = load_csv(csv_path)
    ensure_column_exists(df, label_col)

    label_str = _series_to_str(df[label_col])
    counts = label_str.value_counts(dropna=False)

    labels = list(counts.index)
    values = list(counts.values)

    if filename is None:
        filename = f"label_distribution_{label_col}_{plot_type}.png"

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


@eda.command(name="remove-outliers")
@click.option("--csv_path", required=True, type=str, help="Path to the CSV file")
@click.option("--text_col", required=True, type=str, help="Name of the text column")
@click.option(
    "--method",
    type=click.Choice(["iqr", "zscore"]),
    default="iqr",
    show_default=True,
    help="Outlier detection method: iqr (Interquartile Range) or zscore (Z-Score)",
)
@click.option("--output", required=True, type=str, help="Output CSV filename (saved to data/)")
def remove_outliers(
    csv_path: str,
    text_col: str,
    method: str,
    output: str,
):
    """
    Remove statistical outliers from the dataset based on text length.
    
    IQR Method: Removes rows where text length falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    Z-Score Method: Removes rows where |Z-Score| > 3
    """
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    # Calculate text lengths
    text_series = df[text_col].astype(str).fillna("")
    text_lengths = text_series.apply(lambda s: len(s.split()))

    original_count = len(df)

    click.echo(f"Processing: {csv_path}")
    click.echo(f"Text column: {text_col}")
    click.echo(f"Method: {method.upper()}")
    click.echo("---")

    if method == "iqr":
        # IQR Method
        Q1 = text_lengths.quantile(0.25)
        Q3 = text_lengths.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = max(1, Q1 - 1.5 * IQR)  # At least 1 word
        upper_bound = Q3 + 1.5 * IQR

        click.echo(f"Q1 (25th percentile): {Q1:.1f} words")
        click.echo(f"Q3 (75th percentile): {Q3:.1f} words")
        click.echo(f"IQR: {IQR:.1f} words")
        click.echo(f"Lower bound: {lower_bound:.1f} words")
        click.echo(f"Upper bound: {upper_bound:.1f} words")
        click.echo("---")

        # Filter outliers
        mask = (text_lengths >= lower_bound) & (text_lengths <= upper_bound)
        df_clean = df[mask].copy()

    elif method == "zscore":
        # Z-Score Method
        mean_len = text_lengths.mean()
        std_len = text_lengths.std()
        z_scores = np.abs((text_lengths - mean_len) / std_len)

        click.echo(f"Mean length: {mean_len:.2f} words")
        click.echo(f"Std deviation: {std_len:.2f} words")
        click.echo(f"Z-score threshold: 3")
        click.echo("---")

        # Filter outliers (Z-score > 3)
        mask = z_scores <= 3
        df_clean = df[mask].copy()

    outliers_count = original_count - len(df_clean)
    outliers_pct = (outliers_count / original_count * 100) if original_count > 0 else 0

    click.echo(f"Original rows: {original_count:,}")
    click.echo(f"Outliers detected: {outliers_count:,}")
    click.echo(f"Rows kept: {len(df_clean):,}")
    click.echo(f"Outliers removed: {outliers_pct:.1f}%")

    # Save output
    out_path = Path("data") / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False, encoding="utf-8")

    click.echo(f"Saved → {out_path}")

