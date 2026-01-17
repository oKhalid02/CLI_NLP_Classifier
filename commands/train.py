# commands/train.py
from __future__ import annotations

import ast
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.data_handler import load_csv, ensure_column_exists


# -------------------------
# Logging helpers
# -------------------------
def log(msg: str) -> None:
    click.echo(f"[train] {msg}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------
# Model + parsing helpers
# -------------------------
def _is_embeddings_file(s: str) -> bool:
    """Treat input_col as embeddings file if it exists OR looks like a pkl/joblib path."""
    p = Path(s)
    if p.exists() and p.is_file():
        return True
    return p.suffix.lower() in {".pkl", ".pickle", ".joblib"}


def _parse_hparams(spec: str) -> tuple[str, Dict[str, Any]]:
    """
    "lr:C=0.5,max_iter=3000" -> ("lr", {"C":0.5,"max_iter":3000})
    "rf:n_estimators=200" -> ("rf", {"n_estimators":200})
    "knn" -> ("knn", {})
    """
    if ":" not in spec:
        return spec.lower().strip(), {}

    name, params_str = spec.split(":", 1)
    name = name.lower().strip()
    params: Dict[str, Any] = {}

    for part in params_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise click.UsageError(f"Bad hyperparam '{part}' in '{spec}'. Use key=value.")
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()

        if v.lower() in ("true", "false"):
            params[k] = (v.lower() == "true")
        else:
            try:
                params[k] = int(v)
            except ValueError:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v

    return name, params


def _expand_models(models: List[str]) -> List[str]:
    """
    Teacher inputs:
      - [] -> default knn/lr/rf
      - ["all"] -> a bigger set
      - ["knn", "lr", "rf"] -> as-is
      - ["knn:n_neighbors=7", "lr:C=0.5"] -> as-is
    """
    if not models:
        return ["knn", "lr", "rf"]

    if len(models) == 1 and models[0].lower() == "all":
        # IMPORTANT: NB is count-based; embeddings can be negative. We'll skip NB later if invalid.
        return ["knn", "lr", "rf", "svm", "nb", "dt"]

    return models


def _make_model(name: str, params: Dict[str, Any]):
    name = name.lower()

    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**params)

    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        default = {"max_iter": 3000}
        default.update(params)
        return LogisticRegression(**default)

    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        default = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
        default.update(params)
        return RandomForestClassifier(**default)

    if name == "svm":
        from sklearn.svm import LinearSVC
        default = {"random_state": 42}
        default.update(params)
        return LinearSVC(**default)

    if name == "nb":
        # MultinomialNB requires non-negative features (counts/TF-IDF).
        from sklearn.naive_bayes import MultinomialNB
        return MultinomialNB(**params)

    if name == "dt":
        from sklearn.tree import DecisionTreeClassifier
        default = {"random_state": 42}
        default.update(params)
        return DecisionTreeClassifier(**default)

    raise click.UsageError(f"Unknown model '{name}'. Use knn, lr, rf, svm, nb, dt, or all.")


# -------------------------
# Embedding loaders
# -------------------------
def _load_embeddings_from_pkl(path: str) -> np.ndarray:
    """
    Supports:
      - dict with {"vectors": np.ndarray}
      - raw np.ndarray
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    with open(p, "rb") as f:
        obj = pickle.load(f)

    vecs = obj["vectors"] if isinstance(obj, dict) and "vectors" in obj else obj
    X = np.asarray(vecs)

    if X.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (n_samples, dim). Got shape {X.shape}")
    return X


def _load_embeddings_from_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Embedding column must contain list/tuple per row:
      - "[0.1, 0.2, ...]"
      - "(0.1, 0.2, ...)"
      - or already a python list/np array
    """
    ensure_column_exists(df, col)
    rows: List[List[float]] = []

    for x in df[col]:
        if pd.isna(x):
            rows.append([])
            continue
        if isinstance(x, (list, tuple, np.ndarray)):
            rows.append([float(v) for v in list(x)])
            continue

        s = str(x).strip()
        if not s:
            rows.append([])
            continue

        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                rows.append([float(t) for t in v])
            else:
                rows.append([])
        except Exception:
            rows.append([])

    max_len = max((len(r) for r in rows), default=0)
    if max_len == 0:
        raise click.UsageError(
            f"Could not parse embeddings from column '{col}'. "
            "Expected values like [0.1, 0.2, ...]."
        )

    clean = []
    bad = 0
    for r in rows:
        if len(r) != max_len:
            bad += 1
            clean.append([0.0] * max_len)
        else:
            clean.append(r)

    if bad:
        log(f"Warning: {bad} rows had wrong embedding length; padded with zeros.")

    return np.asarray(clean, dtype=np.float32)


# -------------------------
# Plots (only confusion matrix + ROC bonus)
# -------------------------
def _save_confusion_matrix_plot(cm: np.ndarray, class_names: List[str], out_path: Path, title: str):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _try_save_roc_plot(model, X_test, y_test, out_path: Path, title: str) -> Optional[float]:
    """ROC only if binary and model supports predict_proba or decision_function."""
    if len(np.unique(y_test)) != 2:
        return None

    scores = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)
        scores = prob[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)

    if scores is None:
        return None

    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = float(auc(fpr, tpr))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return roc_auc


# -------------------------
# Teacher-proof CLI command
# -------------------------
@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--csv_path", required=True, type=str, help="Path to CSV")
@click.option(
    "--input_col",
    required=True,
    type=str,
    help="Either embeddings file path (outputs/embeddings/*.pkl) OR embedding column name in CSV",
)
@click.option("--output_col", required=True, type=str, help="Label column in CSV (target y)")
@click.option("--test_size", default=0.2, show_default=True, type=float, help="Test split ratio")
@click.option("--seed", default=42, show_default=True, type=int, help="Random seed")
@click.option(
    "--models",
    required=False,
    type=str,
    default=None,
    help='Teacher style: --models knn lr rf (extra tokens captured) OR --models all OR --models "lr:C=0.5" ...',
)
@click.option("--save_model", default=None, type=str, help="Optional path to save best model")
@click.pass_context
def train(
    ctx: click.Context,
    csv_path: str,
    input_col: str,
    output_col: str,
    test_size: float,
    seed: int,
    models: Optional[str],
    save_model: Optional[str],
):
    """
    Works with teacher commands like:
      python main.py train --csv_path final.csv --input_col outputs/embeddings/embeded_vec.pkl --output_col class --models knn lr rf
      python main.py train --csv_path final.csv --input_col outputs/embeddings/embeded_vec.pkl --output_col class --models all
      python main.py train --csv_path final.csv --input_col outputs/embeddings/embeded_vec.pkl --output_col class --models "knn:n_neighbors=7" "lr:C=0.5"
    """

    # -------- Parse models in teacher style --------
    extra_tokens = list(ctx.args)  # tokens after known options
    model_tokens: List[str]
    if models is None:
        model_tokens = []
    else:
        model_tokens = [models] + extra_tokens

    model_list = _expand_models(model_tokens)

    log(f"CSV: {csv_path}")
    log(f"Embeddings source (--input_col): {input_col}")
    log(f"Label column (--output_col): {output_col}")
    log(f"Models requested: {model_tokens if model_tokens else '[default knn lr rf]'}")
    log(f"Models expanded: {model_list}")

    # -------- Load CSV --------
    df = load_csv(csv_path)
    log(f"Loaded CSV rows={len(df)} cols={list(df.columns)}")
    ensure_column_exists(df, output_col)

    # -------- Load embeddings --------
    if _is_embeddings_file(input_col):
        log(f"Loading embeddings from file: {input_col}")
        X = _load_embeddings_from_pkl(input_col)
        log(f"Embeddings shape: {X.shape}")
        if len(X) != len(df):
            raise click.UsageError(
                f"Embeddings rows ({len(X)}) != CSV rows ({len(df)}). "
                "Make sure embeddings were created from the SAME CSV (same order)."
            )
        embeddings_source = f"file:{input_col}"
    else:
        log(f"Loading embeddings from CSV column: {input_col}")
        ensure_column_exists(df, input_col)
        X = _load_embeddings_from_column(df, input_col)
        log(f"Embeddings shape: {X.shape}")
        embeddings_source = f"column:{input_col}"

    # -------- Clean rows: drop missing labels + zero vectors --------
    y_raw = df[output_col]
    keep = ~pd.isna(y_raw)
    keep = keep & (np.abs(X).sum(axis=1) > 0)

    dropped = int((~keep).sum())
    if dropped:
        log(f"Dropping {dropped} rows (missing label or zero-vector)")

    df2 = df[keep].copy()
    X2 = X[keep.values]
    y2 = df2[output_col].astype(str).values

    if len(df2) < 20:
        raise click.UsageError("Not enough clean rows to train after dropping invalid rows.")

    # -------- Encode labels (dynamic classes) --------
    le = LabelEncoder()
    y_enc = le.fit_transform(y2)
    classes = list(le.classes_)
    log(f"Detected classes ({len(classes)}): {classes}")

    # -------- Split HERE --------
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y_enc, test_size=test_size, random_state=seed, stratify=y_enc
    )
    log(f"Split: train={len(y_train)} test={len(y_test)} (test_size={test_size})")

    # -------- Output dirs --------
    ts = _timestamp()
    reports_dir = _ensure_dir("outputs/reports")
    viz_dir = _ensure_dir("outputs/visualizations")
    models_dir = _ensure_dir("outputs/models")

    report_path = reports_dir / f"training_report_{ts}.md"

    # -------- Markdown report header --------
    md: List[str] = []
    md.append(f"# Training Report ({ts})\n")
    md.append("## Dataset & Settings")
    md.append(f"- CSV: `{csv_path}`")
    md.append(f"- Embeddings source: `{embeddings_source}`")
    md.append(f"- Label column: `{output_col}`")
    md.append(f"- Classes ({len(classes)}): {', '.join([f'`{c}`' for c in classes])}")
    md.append(f"- Split: test_size = **{test_size}**, seed = **{seed}**")
    md.append(f"- X shape: `{X2.shape}`\n")

    md.append("## Model Metrics (per model)\n")
    md.append("| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC AUC (binary) |")
    md.append("|---|---:|---:|---:|---:|---:|")

    # For correct image links inside outputs/reports/*.md:
    # images are in outputs/visualizations/*.png, so relative path is "../visualizations/..."
    per_model: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    # Pre-check for NB validity (MultinomialNB requires non-negative values)
    has_negative = bool(np.min(X_train) < 0)

    for spec in model_list:
        name, params = _parse_hparams(spec)
        log(f"Training model: {spec} (parsed name={name}, params={params})")

        # Skip MultinomialNB when embeddings contain negatives
        if name == "nb" and has_negative:
            log("Skipping nb: MultinomialNB requires non-negative features, but embeddings contain negatives.")
            continue

        clf = _make_model(name, params)

        try:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
        except Exception as e:
            log(f"Skipping {spec} due to error: {e}")
            continue

        acc = float(accuracy_score(y_test, pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="macro", zero_division=0
        )
        prec, rec, f1 = float(prec), float(rec), float(f1)

        cm = confusion_matrix(y_test, pred)
        cm_plot_path = viz_dir / f"cm_{name}_{ts}.png"
        _save_confusion_matrix_plot(cm, classes, cm_plot_path, title=f"Confusion Matrix: {spec}")
        log(f"Saved confusion matrix: {cm_plot_path}")

        roc_auc = _try_save_roc_plot(
            clf, X_test, y_test,
            out_path=(viz_dir / f"roc_{name}_{ts}.png"),
            title=f"ROC Curve: {spec}",
        )
        roc_plot_path = (viz_dir / f"roc_{name}_{ts}.png") if roc_auc is not None else None
        if roc_auc is not None:
            log(f"Saved ROC curve: {roc_plot_path} (AUC={roc_auc:.4f})")

        md.append(
            f"| `{spec}` | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | "
            f"{(f'{roc_auc:.4f}' if roc_auc is not None else 'N/A')} |"
        )

        per_model.append({
            "spec": spec,
            "cm_plot": str(cm_plot_path),
            "roc_auc": roc_auc,
            "roc_plot": str(roc_plot_path) if roc_plot_path else None,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
        })

        if best is None or f1 > best["f1_macro"]:
            best = {
                "spec": spec,
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "estimator": clf,
            }

        log(f"Done: {spec} -> acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    if not per_model or best is None:
        raise click.UsageError("No models were successfully trained. Check logs above.")

    # -------- Confusion matrices section (FIXED image paths) --------
    md.append("\n## Confusion Matrices (per model)\n")
    for row in per_model:
        md.append(f"### `{row['spec']}`\n")

        # row['cm_plot'] is "outputs/visualizations/....png"
        cm_rel_from_outputs = Path(row["cm_plot"]).relative_to("outputs")  # "visualizations/....png"
        md.append(f"![Confusion Matrix](../{cm_rel_from_outputs.as_posix()})\n")

        if row["roc_auc"] is not None and row["roc_plot"]:
            roc_rel_from_outputs = Path(row["roc_plot"]).relative_to("outputs")
            md.append(f"**ROC Curve (bonus):** AUC = **{row['roc_auc']:.4f}**\n")
            md.append(f"![ROC Curve](../{roc_rel_from_outputs.as_posix()})\n")

    # -------- Best model --------
    md.append("\n## Best Model\n")
    md.append(f"- Best by **macro F1**: `{best['spec']}`")
    md.append(f"- Accuracy: **{best['accuracy']:.4f}**")
    md.append(f"- Precision (macro): **{best['precision_macro']:.4f}**")
    md.append(f"- Recall (macro): **{best['recall_macro']:.4f}**")
    md.append(f"- F1 (macro): **{best['f1_macro']:.4f}**\n")

    # -------- Save best model --------
    if save_model:
        model_path = Path('outputs/models/' + save_model)
    else:
        model_path = models_dir / f"best_model_{ts}.pkl"
    _ensure_dir(str(model_path.parent))

    payload = {
        "best_model_spec": best["spec"],
        "model": best["estimator"],
        "label_encoder": le,
        "classes": classes,
        "csv_path": csv_path,
        "input_col": input_col,
        "output_col": output_col,
        "test_size": test_size,
        "seed": seed,
        "trained_at": ts,
    }
    dump(payload, model_path)
    log(f"Saved best model: {model_path}")

    md.append(f"- Saved model: `{model_path}`\n")

    # -------- Write report --------
    report_path.write_text("\n".join(md), encoding="utf-8")
    log(f"Saved report: {report_path}")

    click.echo(f"\n✅ Report saved → {report_path}")
    click.echo(f"✅ Best model saved → {model_path}")
