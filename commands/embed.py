from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import numpy as np
import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.data_handler import load_csv, ensure_column_exists


def _ensure_out(path: str) -> Path:
    p = Path(path)
    if p.parent == Path("."):
        p = Path("outputs/embeddings") / p.name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p



def _drop_empty(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df[text_col] = df[text_col].astype(str).fillna("").str.strip()
    return df[df[text_col].str.len() > 0].copy()


def _estimate_size_mb(x: Any) -> float:
    # Sparse matrix
    if sparse.issparse(x):
        # data + indices + indptr (rough, but good)
        size = x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
        return size / (1024**2)

    # Numpy array
    if isinstance(x, np.ndarray):
        return x.nbytes / (1024**2)

    # Fallback: pickle size estimate (can be expensive, but ok for small meta)
    try:
        b = pickle.dumps(x)
        return len(b) / (1024**2)
    except Exception:
        return float("nan")


def _print_stats(name: str, vectors: Any):
    if sparse.issparse(vectors):
        shape = vectors.shape
        nnz = vectors.nnz
        mb = _estimate_size_mb(vectors)
        click.echo(f"{name} shape: {shape} | nnz={nnz} | approx_mem={mb:.2f} MB")
    elif isinstance(vectors, np.ndarray):
        shape = vectors.shape
        mb = _estimate_size_mb(vectors)
        click.echo(f"{name} shape: {shape} | dtype={vectors.dtype} | approx_mem={mb:.2f} MB")
    else:
        click.echo(f"{name}: (unknown type) approx_mem={_estimate_size_mb(vectors):.2f} MB")


@click.group()
def embed():
    """Text embedding commands"""
    pass


@embed.command()
@click.option("--csv_path", required=True, type=str, help="Path to CSV")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--max_features", default=5000, show_default=True, type=int, help="TF-IDF vocabulary size")
@click.option("--ngram_max", default=2, show_default=True, type=int, help="Max n-gram (1..ngram_max)")
@click.option("--min_df", default=2, show_default=True, type=int, help="Ignore very rare terms")
@click.option("--output", required=True, type=str, help="Output .pkl (joblib) path")
def tfidf(csv_path: str, text_col: str, max_features: int, ngram_max: int, min_df: int, output: str):
    """TF-IDF embedding (sklearn)"""
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    df = _drop_empty(df, text_col)
    texts = df[text_col].astype(str).tolist()

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        lowercase=False,  # Arabic: keep as-is
    )
    X = vec.fit_transform(texts)

    _print_stats("TF-IDF vectors", X)

    out = _ensure_out(output)
    payload = {
        "type": "tfidf",
        "csv_path": csv_path,
        "text_col": text_col,
        "n_rows": len(df),
        "vectorizer": vec,
        "vectors": X,
    }
    dump(payload, out)
    click.echo(f"Saved → {out}")


@embed.command()
@click.option("--csv_path", required=True, type=str, help="Path to CSV")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--model", default="JadwalAlmaa/model2vec-ARBERTv2", show_default=True, type=str,
              help="Model2Vec model name on Hugging Face")
@click.option("--batch_size", default=256, show_default=True, type=int, help="Batch size")
@click.option("--output", required=True, type=str, help="Output .pkl path")
def model2vec(csv_path: str, text_col: str, model: str, batch_size: int, output: str):
    """Model2Vec embedding (StaticModel)"""
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    df = _drop_empty(df, text_col)
    texts = df[text_col].astype(str).tolist()

    # Lazy import
    from model2vec import StaticModel

    m = StaticModel.from_pretrained(model)
    vectors = m.encode(texts, batch_size=batch_size)

    # Ensure numpy
    vectors = np.asarray(vectors)

    _print_stats("Model2Vec vectors", vectors)

    out = _ensure_out(output)
    payload = {
        "type": "model2vec",
        "model": model,
        "csv_path": csv_path,
        "text_col": text_col,
        "n_rows": len(df),
        "vectors": vectors,
    }
    with open(out, "wb") as f:
        pickle.dump(payload, f)

    click.echo(f"Saved → {out}")


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # last_hidden: (B, T, H), mask: (B, T)
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


@embed.command()
@click.option("--csv_path", required=True, type=str, help="Path to CSV")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--model", default="aubmindlab/bert-base-arabertv2", show_default=True, type=str,
              help="HF model name (AraBERT example)")
@click.option("--pooling", type=click.Choice(["mean", "cls"]), default="mean", show_default=True,
              help="Pooling strategy")
@click.option("--max_length", default=256, show_default=True, type=int, help="Tokenizer max length")
@click.option("--batch_size", default=16, show_default=True, type=int, help="Batch size")
@click.option("--device", default="auto", show_default=True, type=str, help="auto|cpu|cuda|mps")
@click.option("--output", required=True, type=str, help="Output .pkl path")
def bert(
    csv_path: str,
    text_col: str,
    model: str,
    pooling: str,
    max_length: int,
    batch_size: int,
    device: str,
    output: str,
):
    """BERT embeddings (transformers)"""
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    df = _drop_empty(df, text_col)
    texts = df[text_col].astype(str).tolist()

    import torch
    from transformers import AutoTokenizer, AutoModel

    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModel.from_pretrained(model)
    mdl.eval()
    mdl.to(dev)

    all_vecs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(dev)

            out = mdl(**enc)
            last_hidden = out.last_hidden_state  # (B,T,H)

            if pooling == "cls":
                vec = last_hidden[:, 0, :]
            else:
                # mean pooling with attention mask
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                vec = summed / counts

            all_vecs.append(vec.detach().cpu().numpy())

    vectors = np.vstack(all_vecs)
    _print_stats("BERT vectors", vectors)

    outp = _ensure_out(output)
    payload = {
        "type": "bert",
        "model": model,
        "pooling": pooling,
        "csv_path": csv_path,
        "text_col": text_col,
        "n_rows": len(df),
        "vectors": vectors,
    }
    with open(outp, "wb") as f:
        pickle.dump(payload, f)

    click.echo(f"Saved → {outp}")


@embed.command(name="sentence-transformer")
@click.option("--csv_path", required=True, type=str, help="Path to CSV")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--model", default="sentence-transformers/distiluse-base-multilingual-cased-v2",
              show_default=True, type=str, help="Sentence-Transformers model name")
@click.option("--batch_size", default=32, show_default=True, type=int, help="Batch size")
@click.option("--device", default="auto", show_default=True, type=str, help="auto|cpu|cuda|mps")
@click.option("--output", required=True, type=str, help="Output .pkl path")
def sentence_transformer(csv_path: str, text_col: str, model: str, batch_size: int, device: str, output: str):
    """Sentence Transformers embeddings"""
    df = load_csv(csv_path)
    ensure_column_exists(df, text_col)

    df = _drop_empty(df, text_col)
    texts = df[text_col].astype(str).tolist()

    from sentence_transformers import SentenceTransformer

    if device == "auto":
        # SentenceTransformer accepts "cuda", "cpu", and sometimes "mps" depending on torch build
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = device

    st = SentenceTransformer(model, device=dev)
    vectors = st.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    _print_stats("SentenceTransformer vectors", vectors)

    outp = _ensure_out(output)
    payload = {
        "type": "sentence-transformer",
        "model": model,
        "csv_path": csv_path,
        "text_col": text_col,
        "n_rows": len(df),
        "vectors": vectors,
    }
    with open(outp, "wb") as f:
        pickle.dump(payload, f)

    click.echo(f"Saved → {outp}")
