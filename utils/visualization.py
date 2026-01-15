from __future__ import annotations

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt


def _ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pie_chart(
    labels: Sequence[str],
    values: Sequence[int],
    title: str,
    out_dir: str,
    filename: str,
) -> str:
    out_path = _ensure_out_dir(out_dir) / filename

    plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return str(out_path)


def save_bar_chart(
    labels: Sequence[str],
    values: Sequence[int],
    title: str,
    xlabel: str,
    ylabel: str,
    out_dir: str,
    filename: str,
) -> str:
    out_path = _ensure_out_dir(out_dir) / filename

    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return str(out_path)


def save_histogram(
    values: Sequence[int],
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    out_dir: str,
    filename: str,
) -> str:
    out_path = _ensure_out_dir(out_dir) / filename

    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return str(out_path)
