"""
EduPredict - Student Performance Analytics
Student Performance Prediction Using Decision Tree Algorithm (Entropy / Information Gain)

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
import math
import re
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from generate_dataset import generate_students

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
APP_NAME = "EduPredict – Student Performance Analytics"
TARGET = "Performance"
ID_COL = "Student_ID"
NUMERIC_FEATURES = [
    "Attendance",
    "Previous_Exam_Marks",
    "Assignment_Marks",
    "Internal_Marks",
    "Study_Hours",
    "Assignment_Completion",
]
CATEGORICAL_FEATURES = ["Class_Participation"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PARTICIPATION_LEVELS = ["Low", "Medium", "High"]
CLASS_ORDER = ["Low", "Medium", "High"]
VALUE_RANGES = {
    "Attendance": (0.0, 100.0),
    "Previous_Exam_Marks": (0.0, 100.0),
    "Assignment_Marks": (0.0, 100.0),
    "Internal_Marks": (0.0, 100.0),
    "Study_Hours": (0.0, 24.0),
    "Assignment_Completion": (0.0, 100.0),
}
LABELS = {
    "Attendance": "Attendance (%)",
    "Previous_Exam_Marks": "Previous Exam Marks",
    "Assignment_Marks": "Assignment Marks",
    "Internal_Marks": "Internal Marks",
    "Study_Hours": "Study Hours / Day",
    "Class_Participation": "Class Participation",
    "Assignment_Completion": "Assignment Completion (%)",
}
SHORT = {c: c.replace("_", " ") for c in ALL_FEATURES}

CLASS_COLORS = {"Low": "#e5534b", "Medium": "#f2a33a", "High": "#2e9e6b"}
BLUE = "#1d4ed8"
DARK_BLUE = "#0f2a5c"
GREY = "#64748b"

MIN_ROWS = 20
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_PARAMS = {"limit_depth": True, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5}

PAGES = [
    "Dashboard",
    "Individual Student Prediction",
    "Dataset Management",
    "Model Training",
    "Model Evaluation",
    "Decision Tree Visualization",
    "Batch Prediction",
    "About Project",
]

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": "#cbd5e1",
        "grid.color": "#e5e9f0",
    }
)

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1300px;}
.app-banner {background: linear-gradient(120deg, #0f2a5c 0%, #1d4ed8 100%); color: #ffffff; padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;}
.app-banner h1 {color: #ffffff; font-size: 1.7rem; margin: 0 0 4px 0; padding: 0;}
.app-banner p {color: #dbe7ff; margin: 0; font-size: 0.95rem;}
.metric-card {background: #ffffff; border: 1px solid #e3e8f0; border-left: 5px solid #1d4ed8; border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,24,40,0.06); margin-bottom: 12px;}
.metric-label {color: #64748b; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;}
.metric-value {color: #0f2a5c; font-size: 1.85rem; font-weight: 700; line-height: 1.25;}
.metric-sub {color: #94a3b8; font-size: 0.78rem; min-height: 1rem;}
.section-title {color: #0f2a5c; font-size: 1.15rem; font-weight: 700; margin: 18px 0 6px 0; border-bottom: 2px solid #dbe7ff; padding-bottom: 4px;}
.result-card {background: #ffffff; border: 2px solid #1d4ed8; border-radius: 14px; padding: 22px; text-align: center; box-shadow: 0 2px 8px rgba(16,24,40,0.08);}
.result-title {color: #64748b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;}
.result-value {font-size: 2.6rem; font-weight: 800; line-height: 1.2;}
.result-sub {color: #64748b; font-size: 0.9rem;}
.prob-row {display: flex; align-items: center; gap: 10px; margin: 10px 0;}
.prob-name {width: 70px; font-weight: 600; color: #1f2937;}
.prob-track {flex: 1; background: #e8edf5; border-radius: 8px; height: 16px; overflow: hidden;}
.prob-fill {height: 16px; border-radius: 8px;}
.prob-val {width: 60px; text-align: right; font-weight: 600; color: #1f2937;}
.brand {padding: 6px 0 10px 0;}
.brand-title {color: #0f2a5c; font-size: 1.25rem; font-weight: 800; line-height: 1.2;}
.brand-sub {color: #64748b; font-size: 0.78rem;}
.note-box {background: #eef4ff; border: 1px solid #c7d7fb; color: #1e3a8a; border-radius: 10px; padding: 10px 14px; font-size: 0.9rem;}
</style>
"""


# --------------------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------------------
def render_html(markup: str) -> None:
    """Render HTML on a single line (avoids Markdown turning indented lines into code)."""
    st.markdown(" ".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    render_html(f'<div class="app-banner"><h1>{title}</h1><p>{subtitle}</p></div>')


def section(title: str, caption: Optional[str] = None) -> None:
    render_html(f'<div class="section-title">{title}</div>')
    if caption:
        st.caption(caption)


def metric_card(label: str, value: str, sub: str = "", accent: str = BLUE) -> str:
    return (
        f'<div class="metric-card" style="border-left-color:{accent}">'
        f'<div class="metric-label">{label}</div><div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>'
    )


def metric_row(cards: List[str]) -> None:
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(card, unsafe_allow_html=True)


def show_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def _drop_legend(ax) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------------------
# Data validation and cleaning
# --------------------------------------------------------------------------------------
class DatasetError(Exception):
    """Raised for problems with a dataset that the user can understand and fix."""


def _canon(name: object) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip()).lower()


_CANON_MAP = {_canon(c): c for c in ALL_FEATURES + [TARGET, ID_COL]}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns such as 'previous exam marks' or ' attendance ' to the expected names."""
    rename: Dict[str, str] = {}
    for col in df.columns:
        key = _canon(col)
        if key in _CANON_MAP and _CANON_MAP[key] not in rename.values():
            rename[col] = _CANON_MAP[key]
    return df.rename(columns=rename)


def _norm_level(value: object) -> object:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    text = str(value).strip().title()
    return text if text in PARTICIPATION_LEVELS else np.nan


def coerce_features(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Convert feature columns to the right types. Invalid values become missing (NaN)."""
    out = df.copy()
    stats = {"non_numeric": 0, "out_of_range": 0, "invalid_category": 0}
    for col in columns:
        if col in NUMERIC_FEATURES:
            was_missing = out[col].isna()
            numbers = pd.to_numeric(out[col], errors="coerce")
            stats["non_numeric"] += int((numbers.isna() & ~was_missing).sum())
            lo, hi = VALUE_RANGES[col]
            bad = numbers.notna() & ((numbers < lo) | (numbers > hi))
            stats["out_of_range"] += int(bad.sum())
            out[col] = numbers.mask(bad).astype(float)
        elif col in CATEGORICAL_FEATURES:
            raw = out[col]
            cleaned = raw.map(_norm_level)
            stats["invalid_category"] += int((cleaned.isna() & raw.notna()).sum())
            out[col] = cleaned.astype(object)
    return out, stats


def clean_dataset(raw: pd.DataFrame, remove_duplicates: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Validate and clean an uploaded dataset. Raises DatasetError with a clear message."""
    if raw is None or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise DatasetError("The dataset is empty. Please upload a CSV file that contains student records.")

    df = standardize_columns(raw)
    missing_cols = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise DatasetError(
            "The dataset is missing required column(s): "
            + ", ".join(missing_cols)
            + ". Expected columns: "
            + ", ".join(ALL_FEATURES + [TARGET])
            + ". Download the CSV template from the Dataset Management page."
        )

    report: Dict = {"raw_shape": tuple(raw.shape)}
    keep = ([ID_COL] if ID_COL in df.columns else []) + ALL_FEATURES + [TARGET]
    report["ignored_columns"] = [c for c in df.columns if c not in keep]
    df = df[keep].copy()
    report["missing_before"] = {k: int(v) for k, v in df.isna().sum().items()}

    df, stats = coerce_features(df, ALL_FEATURES)
    report.update(stats)

    target = df[TARGET].map(lambda v: str(v).strip().title() if pd.notna(v) else np.nan)
    invalid_target = ~target.isin(CLASS_ORDER)
    report["invalid_target_rows"] = int(invalid_target.sum())
    df[TARGET] = target
    df = df[~invalid_target]

    dup_mask = df.duplicated(subset=ALL_FEATURES + [TARGET], keep="first")
    report["duplicates_found"] = int(dup_mask.sum())
    report["duplicates_removed"] = 0
    if remove_duplicates and dup_mask.any():
        df = df[~dup_mask]
        report["duplicates_removed"] = int(dup_mask.sum())

    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise DatasetError(
            f"No valid rows remain. The '{TARGET}' column must contain only Low, Medium or High values."
        )
    if ID_COL not in df.columns:
        df.insert(0, ID_COL, [f"STU{i + 1:04d}" for i in range(len(df))])
        report["id_generated"] = True
    else:
        df[ID_COL] = df[ID_COL].astype(str)
        report["id_generated"] = False

    empty_cols = [c for c in ALL_FEATURES if df[c].notna().sum() == 0]
    if empty_cols:
        raise DatasetError("These columns contain no valid values: " + ", ".join(empty_cols))
    if df[TARGET].nunique() < 2:
        raise DatasetError(
            f"The '{TARGET}' column contains only one class ({df[TARGET].iloc[0]}). At least two classes are needed."
        )

    report["final_shape"] = tuple(df.shape)
    report["missing_after"] = {k: int(v) for k, v in df[ALL_FEATURES].isna().sum().items()}
    report["class_counts"] = df[TARGET].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_dict()
    return df, report


def read_csv_upload(uploaded) -> pd.DataFrame:
    data = uploaded.getvalue()
    if not data.strip():
        raise DatasetError("The uploaded file is empty.")
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(data), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise DatasetError("The uploaded file contains no data.")
        except pd.errors.ParserError as exc:
            raise DatasetError(f"The file could not be parsed as a CSV: {exc}")
        except Exception as exc:  # noqa: BLE001
            raise DatasetError(f"The file could not be read as a CSV: {exc}")
    else:
        raise DatasetError("The file encoding could not be recognised.")
    if frame.shape[0] == 0:
        raise DatasetError("The CSV has column headers but no records.")
    return frame


def template_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: ["STU0001", "STU0002", "STU0003"],
            "Attendance": [92.0, 74.5, 55.0],
            "Previous_Exam_Marks": [85.0, 66.0, 42.0],
            "Assignment_Marks": [88.0, 70.0, 48.0],
            "Internal_Marks": [90.0, 68.0, 45.0],
            "Study_Hours": [4.5, 3.0, 1.0],
            "Class_Participation": ["High", "Medium", "Low"],
            "Assignment_Completion": [98.0, 80.0, 50.0],
            TARGET: ["High", "Medium", "Low"],
        }
    )


# --------------------------------------------------------------------------------------
# Machine learning pipeline
# --------------------------------------------------------------------------------------
def to_object_array(X) -> np.ndarray:
    """Convert categorical columns to an object array with np.nan for missing values."""
    frame = pd.DataFrame(X).astype(object)
    return frame.where(frame.notna(), np.nan).to_numpy()


def build_pipeline(features: List[str], params: Dict) -> Tuple[Pipeline, List[str]]:
    num = [c for c in NUMERIC_FEATURES if c in features]
    cat = [c for c in CATEGORICAL_FEATURES if c in features]
    transformers = []
    if num:
        transformers.append(("num", SimpleImputer(strategy="median", keep_empty_features=True), num))
    if cat:
        cat_pipe = Pipeline(
            [
                ("to_object", FunctionTransformer(to_object_array)),
                ("impute", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
                ("encode", OrdinalEncoder(categories=[PARTICIPATION_LEVELS] * len(cat))),
            ]
        )
        transformers.append(("cat", cat_pipe, cat))
    prep = ColumnTransformer(transformers, remainder="drop")
    max_depth = int(params["max_depth"]) if params["limit_depth"] else None
    tree = DecisionTreeClassifier(
        criterion="entropy",
        random_state=RANDOM_STATE,
        max_depth=max_depth,
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
    )
    return Pipeline([("prep", prep), ("model", tree)]), num + cat


def train_model(df: pd.DataFrame, features: List[str], params: Dict) -> Dict:
    """Train the Decision Tree pipeline and evaluate it on the held-out test split."""
    if not features:
        raise DatasetError("Select at least one feature before training.")
    if not any(f in NUMERIC_FEATURES for f in features):
        raise DatasetError("Select at least one numeric feature (the model needs more than participation alone).")
    n = len(df)
    if n < MIN_ROWS:
        raise DatasetError(f"At least {MIN_ROWS} valid records are required to train a model (found {n}).")
    y = df[TARGET]
    n_classes = y.nunique()
    if n_classes < 2:
        raise DatasetError("The dataset contains only one performance class, so a classifier cannot be trained.")

    pipeline, ordered = build_pipeline(features, params)
    X = df[ordered]

    n_test = math.ceil(TEST_SIZE * n)
    counts = y.value_counts()
    stratified = bool(counts.min() >= 2 and n_test >= n_classes and (n - n_test) >= n_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if stratified else None
    )
    if y_train.nunique() < 2:
        raise DatasetError("The training split contains only one class. Please provide more varied data.")

    pipeline.fit(X_train, y_train)
    model: DecisionTreeClassifier = pipeline.named_steps["model"]
    y_pred = pipeline.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    rep = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, target_names=CLASS_ORDER, output_dict=True, zero_division=0
    )
    rep.pop("accuracy", None)
    report_df = pd.DataFrame(rep).T
    report_df["support"] = report_df["support"].astype(int)
    report_df.columns = ["Precision", "Recall", "F1-score", "Support"]

    present = set(y_test.unique())
    return {
        "pipeline": pipeline,
        "features": ordered,
        "feature_names": list(ordered),
        "params": dict(params),
        "stratified": stratified,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_index": list(X_train.index),
        "test_index": list(X_test.index),
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "train_accuracy": float(accuracy_score(y_train, pipeline.predict(X_train))),
        "confusion": cm,
        "report_df": report_df,
        "missing_test_classes": [c for c in CLASS_ORDER if c not in present],
        "importances": pd.Series(model.feature_importances_, index=ordered),
        "depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
    }


def predict_frame(bundle: Dict, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Predict classes and class probabilities (columns always Low / Medium / High)."""
    pipe: Pipeline = bundle["pipeline"]
    X = X[bundle["features"]]
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    proba_df = pd.DataFrame(0.0, index=range(len(X)), columns=CLASS_ORDER)
    for i, cls in enumerate(pipe.named_steps["model"].classes_):
        proba_df[cls] = proba[:, i]
    return pred, proba_df


def validate_student_input(values: Dict) -> List[str]:
    errors = []
    for feat in NUMERIC_FEATURES:
        val = values.get(feat)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            errors.append(f"{LABELS[feat]} is required.")
            continue
        lo, hi = VALUE_RANGES[feat]
        if not lo <= float(val) <= hi:
            errors.append(f"{LABELS[feat]} must be between {lo:g} and {hi:g} (you entered {val}).")
    if values.get("Class_Participation") not in PARTICIPATION_LEVELS:
        errors.append("Class Participation must be Low, Medium or High.")
    return errors


# --------------------------------------------------------------------------------------
# Explainability helpers
# --------------------------------------------------------------------------------------
def split_question(feature: str, threshold: float) -> str:
    if feature in CATEGORICAL_FEATURES:
        left = [lv for i, lv in enumerate(PARTICIPATION_LEVELS) if i <= threshold]
        return f"Is {SHORT[feature]} {' or '.join(left)}?"
    return f"Is {SHORT[feature]} ≤ {threshold:.2f}?"


def explain_path(bundle: Dict, input_row: Dict) -> Dict:
    """Follow one student through the trained tree and describe every decision."""
    pipe: Pipeline = bundle["pipeline"]
    model: DecisionTreeClassifier = pipe.named_steps["model"]
    X = pd.DataFrame([input_row])[bundle["features"]]
    Xt = pipe.named_steps["prep"].transform(X)
    tree = model.tree_
    names = bundle["feature_names"]
    node_ids = list(model.decision_path(Xt).indices)
    leaf = node_ids[-1]
    n = tree.weighted_n_node_samples
    imp = tree.impurity
    total = n[0]

    steps, contribution = [], {}
    for node in node_ids[:-1]:
        f = int(tree.feature[node])
        thr = float(tree.threshold[node])
        left, right = tree.children_left[node], tree.children_right[node]
        value = float(Xt[0, f])
        went_left = value <= thr
        info_gain = imp[node] - (n[left] * imp[left] + n[right] * imp[right]) / n[node]
        weighted_gain = (n[node] * imp[node] - n[left] * imp[left] - n[right] * imp[right]) / total
        name = names[f]
        contribution[name] = contribution.get(name, 0.0) + float(weighted_gain)
        if name in CATEGORICAL_FEATURES:
            shown = PARTICIPATION_LEVELS[int(round(value))]
        else:
            shown = f"{value:.1f}"
        steps.append(
            {
                "Step": len(steps) + 1,
                "Question asked by the tree": split_question(name, thr),
                "Student's value": shown,
                "Answer": "Yes" if went_left else "No",
                "Information gain (bits)": round(float(info_gain), 3),
                "Training students at this node": int(round(n[node])),
            }
        )
    return {
        "steps": steps,
        "contribution": contribution,
        "leaf_samples": int(round(n[leaf])),
        "leaf_entropy": float(imp[leaf]),
    }


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def fig_pie(df: pd.DataFrame):
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER).fillna(0)
    counts = counts[counts > 0]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.pie(
        counts.values,
        labels=counts.index,
        colors=[CLASS_COLORS[c] for c in counts.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9},
    )
    ax.set_title("Student Performance Distribution")
    fig.tight_layout()
    return fig


def fig_attendance(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    sns.histplot(
        data=df, x="Attendance", hue=TARGET, hue_order=CLASS_ORDER, palette=CLASS_COLORS,
        multiple="stack", bins=15, edgecolor="white", ax=ax,
    )
    ax.set_title("Attendance Distribution")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Students")
    fig.tight_layout()
    return fig


def fig_study_scatter(df: pd.DataFrame):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for i, cls in enumerate(CLASS_ORDER):
        part = df[df[TARGET] == cls].dropna(subset=["Study_Hours"])
        jitter = rng.uniform(-0.22, 0.22, len(part))
        ax.scatter(part["Study_Hours"], np.full(len(part), i) + jitter, s=18, alpha=0.7,
                   color=CLASS_COLORS[cls], label=cls, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Performance")
    ax.set_title("Study Hours vs Performance")
    fig.tight_layout()
    return fig


def fig_assignment_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    sns.boxplot(
        data=df, x=TARGET, y="Assignment_Marks", order=CLASS_ORDER, hue=TARGET, hue_order=CLASS_ORDER,
        palette=CLASS_COLORS, dodge=False, ax=ax, linewidth=1,
    )
    _drop_legend(ax)
    ax.set_title("Assignment Marks vs Performance")
    ax.set_xlabel("Performance")
    ax.set_ylabel("Assignment marks")
    fig.tight_layout()
    return fig


def fig_internal_hist(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    sns.histplot(df["Internal_Marks"].dropna(), bins=15, kde=True, color=BLUE, edgecolor="white", ax=ax)
    ax.set_title("Internal Marks Distribution")
    ax.set_xlabel("Internal marks")
    ax.set_ylabel("Students")
    fig.tight_layout()
    return fig


def fig_confusion(cm: np.ndarray, normalise: bool):
    fig, ax = plt.subplots(figsize=(5, 4))
    if normalise:
        sums = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm, sums, out=np.zeros(cm.shape, dtype=float), where=sums != 0)
        sns.heatmap(data, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax, linewidths=1, linecolor="white")
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_ORDER,
                    yticklabels=CLASS_ORDER, ax=ax, linewidths=1, linecolor="white")
    ax.set_xlabel("Predicted performance")
    ax.set_ylabel("Actual performance")
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    return fig


def fig_importance(series: pd.Series, title: str = "Feature Importance (Information Gain)"):
    data = series.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh([SHORT.get(i, i) for i in data.index], data.values, color=BLUE)
    for y, v in enumerate(data.values):
        ax.text(v + 0.005, y, f"{v:.1%}", va="center", fontsize=8, color="#1f2937")
    ax.set_xlim(0, max(float(data.max()) * 1.2, 0.05))
    ax.set_xlabel("Share of total impurity reduction")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_tree(bundle: Dict, display_depth: int):
    model: DecisionTreeClassifier = bundle["pipeline"].named_steps["model"]
    width = min(40, max(14, 2.3 * 2 ** display_depth))
    height = max(7, 2.3 * (display_depth + 1))
    fig, ax = plt.subplots(figsize=(width, height))
    plot_tree(
        model,
        feature_names=[SHORT[f] for f in bundle["feature_names"]],
        class_names=[str(c) for c in model.classes_],
        filled=True, rounded=True, impurity=True, precision=2, fontsize=9,
        max_depth=display_depth, ax=ax,
    )
    return fig


# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
def retrain() -> None:
    ss = st.session_state
    try:
        ss.model = train_model(ss.df, ss.features, ss.params)
        ss.train_error = None
    except DatasetError as exc:
        ss.model, ss.train_error = None, str(exc)
    except Exception as exc:  # noqa: BLE001
        ss.model, ss.train_error = None, f"Unexpected error while training the model: {exc}"
    ss.last_prediction = None


def set_dataset(df: pd.DataFrame, report: Dict, name: str, synthetic: bool) -> None:
    ss = st.session_state
    ss.df, ss.report, ss.source_name, ss.is_synthetic = df, report, name, synthetic
    ss.batch_result = None
    retrain()


def load_sample(n_records: int = 300) -> None:
    raw = generate_students(n_records, seed=42)
    df, report = clean_dataset(raw)
    set_dataset(df, report, f"Built-in synthetic sample ({len(df)} records)", True)


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("params", dict(DEFAULT_PARAMS))
    ss.setdefault("features", list(ALL_FEATURES))
    ss.setdefault("last_prediction", None)
    ss.setdefault("batch_result", None)
    ss.setdefault("batch_name", "")
    if "df" not in ss:
        load_sample()


def require_model() -> bool:
    ss = st.session_state
    if ss.get("model") is None:
        st.error(ss.get("train_error") or "No trained model is available yet.")
        st.info("Load a valid dataset in **Dataset Management**, then train the model in **Model Training**.")
        return False
    return True


def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        render_html(
            '<div class="brand"><div class="brand-title">🎓 EduPredict</div>'
            '<div class="brand-sub">Student Performance Analytics</div></div>'
        )
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.caption("Current dataset")
        st.markdown(f"**{ss.source_name}**")
        st.caption(f"{len(ss.df)} records")
        if ss.get("model") is not None:
            st.caption(f"Model test accuracy: **{ss.model['metrics']['accuracy']:.1%}**")
        else:
            st.caption("Model: not trained")


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def page_dashboard() -> None:
    ss = st.session_state
    df = ss.df
    page_header("Dashboard", "Overview of the current dataset – all numbers and charts are computed live from your data.")
    if ss.is_synthetic:
        st.info("This dataset is **synthetic** (randomly generated for demonstration). It does not contain real student records.")

    total = len(df)
    counts = df[TARGET].value_counts()

    def share(cls: str) -> str:
        return f"{counts.get(cls, 0) / total:.1%} of students"

    metric_row(
        [
            metric_card("Total Students", f"{total:,}", f"Source: {ss.source_name[:34]}"),
            metric_card("High Performance", f"{counts.get('High', 0):,}", share("High"), CLASS_COLORS["High"]),
            metric_card("Medium Performance", f"{counts.get('Medium', 0):,}", share("Medium"), CLASS_COLORS["Medium"]),
        ]
    )
    metric_row(
        [
            metric_card("Low Performance", f"{counts.get('Low', 0):,}", share("Low"), CLASS_COLORS["Low"]),
            metric_card("Average Attendance", f"{df['Attendance'].mean():.1f}%", "Across all students"),
            metric_card("Average Study Hours", f"{df['Study_Hours'].mean():.2f} h", "Per day"),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        show_fig(fig_pie(df))
    with c2:
        show_fig(fig_attendance(df))
    with c3:
        show_fig(fig_study_scatter(df))
    c4, c5, _ = st.columns(3)
    with c4:
        show_fig(fig_assignment_box(df))
    with c5:
        show_fig(fig_internal_hist(df))


def _prob_bars(proba: Dict[str, float]) -> str:
    rows = ""
    for cls in CLASS_ORDER:
        pct = proba[cls] * 100
        rows += (
            f'<div class="prob-row"><span class="prob-name">{cls}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%;background:{CLASS_COLORS[cls]}"></div></div>'
            f'<span class="prob-val">{pct:.1f}%</span></div>'
        )
    return rows


def page_prediction() -> None:
    ss = st.session_state
    page_header("Individual Student Prediction", "Enter one student's details and let the trained Decision Tree predict their performance.")
    if not require_model():
        return
    bundle = ss.model

    section("Student Details")
    with st.form("student_form"):
        left, right = st.columns(2)
        with left:
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0, 0.5)
            previous = st.number_input("Previous Exam Marks (0–100)", 0.0, 100.0, 65.0, 0.5)
            assignment = st.number_input("Assignment Marks (0–100)", 0.0, 100.0, 68.0, 0.5)
            internal = st.number_input("Internal Marks (0–100)", 0.0, 100.0, 70.0, 0.5)
        with right:
            study = st.number_input("Study Hours (per day)", 0.0, 24.0, 3.0, 0.25)
            participation = st.selectbox("Class Participation", PARTICIPATION_LEVELS, index=1)
            completion = st.number_input("Assignment Completion (%)", 0.0, 100.0, 85.0, 0.5)
        submitted = st.form_submit_button("Predict Performance", type="primary")

    if submitted:
        values = {
            "Attendance": attendance,
            "Previous_Exam_Marks": previous,
            "Assignment_Marks": assignment,
            "Internal_Marks": internal,
            "Study_Hours": study,
            "Class_Participation": participation,
            "Assignment_Completion": completion,
        }
        errors = validate_student_input(values)
        if errors:
            ss.last_prediction = None
            for message in errors:
                st.error(message)
        else:
            try:
                pred, proba = predict_frame(bundle, pd.DataFrame([values]))
                ss.last_prediction = {"input": values, "pred": str(pred[0]), "proba": proba.iloc[0].to_dict()}
            except Exception as exc:  # noqa: BLE001
                ss.last_prediction = None
                st.error(f"Prediction failed: {exc}")

    result = ss.last_prediction
    if result is None:
        st.info("Fill in the form and press **Predict Performance** to see the result.")
        return

    pred, proba, values = result["pred"], result["proba"], result["input"]
    color = CLASS_COLORS[pred]
    section("Prediction Result")
    left, right = st.columns([1, 1.4])
    with left:
        render_html(
            f'<div class="result-card" style="border-color:{color}"><div class="result-title">Predicted performance</div>'
            f'<div class="result-value" style="color:{color}">{pred}</div>'
            f'<div class="result-sub">Model confidence: {proba[pred]:.1%}</div></div>'
        )
    with right:
        st.markdown("**Prediction probabilities**")
        render_html(_prob_bars(proba))

    imp = bundle["importances"].sort_values(ascending=False)
    top = imp[imp > 0].head(3)
    if len(top):
        st.markdown("**Most influential features in this model**")
        for feat, share_value in top.items():
            st.markdown(f"- **{LABELS[feat]}** – {share_value:.0%} of the tree's total information gain (this student: `{values[feat]}`)")

    section("Why This Prediction?", "A plain-language look at the path this student took through the trained tree.")
    info = explain_path(bundle, values)
    steps = info["steps"]
    if not steps:
        st.info("The trained tree has no splits (it predicts one class for everyone). Try retraining with a smaller minimum-samples setting.")
        return

    best = max(steps, key=lambda s: s["Information gain (bits)"])
    st.markdown(
        f"The Decision Tree asks a series of yes/no questions that it learned from the training data. "
        f"This student answered **{len(steps)} question(s)** and ended in a group (leaf) of "
        f"**{info['leaf_samples']} training students** that is predicted as **{pred}** "
        f"({proba[pred]:.0%} of that group). "
        f"The most informative question on this path was *\"{best['Question asked by the tree']}\"*."
    )
    st.dataframe(pd.DataFrame(steps), hide_index=True)

    contrib = pd.Series(info["contribution"])
    if contrib.sum() > 0:
        contrib = contrib / contrib.sum()
        left, right = st.columns(2)
        with left:
            show_fig(fig_importance(contrib, "Features that mattered on this student's path"))
        with right:
            st.markdown("**How to read this**")
            st.markdown(
                "- The bar chart shows which features were used on **this student's** path, weighted by information gain.\n"
                "- Features that never appear on the path did not influence this particular prediction.\n"
                "- These are patterns the model found in the training data. They show what the tree *used*, "
                "**not** what *causes* good or poor performance."
            )


def _overview_tab() -> None:
    ss = st.session_state
    df, rep = ss.df, ss.report
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Rows", f"{df.shape[0]:,}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Columns", f"{df.shape[1]}"), unsafe_allow_html=True)
    c3.markdown(metric_card("Missing cells", f"{int(df[ALL_FEATURES].isna().sum().sum())}", "Imputed during training"), unsafe_allow_html=True)
    c4.markdown(metric_card("Duplicates removed", f"{rep.get('duplicates_removed', 0)}", f"{rep.get('duplicates_found', 0)} found"), unsafe_allow_html=True)

    section("Cleaning Report")
    lines = [
        f"Original file: **{rep['raw_shape'][0]} rows × {rep['raw_shape'][1]} columns** → processed dataset: **{rep['final_shape'][0]} rows × {rep['final_shape'][1]} columns**",
        f"Rows dropped because `{TARGET}` was missing or not Low/Medium/High: **{rep['invalid_target_rows']}**",
        f"Duplicate records found: **{rep['duplicates_found']}**, removed: **{rep['duplicates_removed']}**",
        f"Non-numeric values converted to missing: **{rep['non_numeric']}**; out-of-range values (marks/percentages outside 0–100, study hours outside 0–24) set to missing: **{rep['out_of_range']}**",
        f"Invalid Class Participation values set to missing: **{rep['invalid_category']}**",
    ]
    if rep.get("id_generated"):
        lines.append(f"No `{ID_COL}` column was found, so IDs were generated automatically.")
    if rep.get("ignored_columns"):
        lines.append("Ignored extra columns: " + ", ".join(f"`{c}`" for c in rep["ignored_columns"]))
    for line in lines:
        st.markdown("- " + line)
    st.markdown(
        '<div class="note-box">Remaining missing values are filled inside the model pipeline – <b>median</b> for numeric columns and '
        '<b>most frequent value</b> for Class Participation – using statistics learned from the <b>training split only</b>, '
        'so no information leaks from the test set.</div>',
        unsafe_allow_html=True,
    )

    section("Columns and Data Types")
    st.dataframe(
        pd.DataFrame(
            {
                "Column": df.columns,
                "Data type": [str(t) for t in df.dtypes],
                "Non-null": df.notna().sum().values,
                "Missing": df.isna().sum().values,
                "Missing %": (df.isna().mean() * 100).round(2).values,
                "Unique values": df.nunique().values,
            }
        ),
        hide_index=True,
    )
    section("Basic Statistics")
    st.dataframe(df[NUMERIC_FEATURES].describe().T.round(2))
    st.markdown("**Class Participation counts**")
    st.dataframe(df["Class_Participation"].value_counts(dropna=False).rename("Students").to_frame())
    section("Dataset Records")
    st.dataframe(df, hide_index=True)


def _records_view() -> pd.DataFrame:
    ss = st.session_state
    view = ss.df.copy()
    bundle = ss.model
    if bundle is not None:
        pred, proba = predict_frame(bundle, view)
        view["Predicted_Performance"] = pred
        view["Confidence_%"] = (proba.max(axis=1).to_numpy() * 100).round(1)
        split = pd.Series("Train", index=view.index)
        split.loc[bundle["test_index"]] = "Test"
        view["Data_Split"] = split
    return view


def _records_tab() -> None:
    ss = st.session_state
    view = _records_view()
    has_pred = "Predicted_Performance" in view.columns
    if has_pred:
        st.caption("Predictions for rows marked 'Train' were seen during training, so they look more accurate than they really are. Use the 'Test' rows for an honest check.")
    else:
        st.warning("No trained model, so predicted performance is not available.")

    f1, f2, f3 = st.columns(3)
    query = f1.text_input("Search by Student ID")
    perf_sel = f2.multiselect("Actual performance", CLASS_ORDER, default=CLASS_ORDER)
    part_sel = f3.multiselect("Class participation", PARTICIPATION_LEVELS, default=PARTICIPATION_LEVELS)
    f4, f5 = st.columns(2)
    pred_sel = f4.multiselect("Predicted performance", CLASS_ORDER, default=CLASS_ORDER, disabled=not has_pred)
    lo, hi = float(np.floor(view["Attendance"].min())), float(np.ceil(view["Attendance"].max()))
    att_range = f5.slider("Attendance range (%)", lo, hi, (lo, hi)) if lo < hi else (lo, hi)

    mask = view["Performance"].isin(perf_sel) & view["Class_Participation"].isin(part_sel)
    mask &= view["Attendance"].between(att_range[0], att_range[1]) | view["Attendance"].isna()
    if has_pred:
        mask &= view["Predicted_Performance"].isin(pred_sel)
    if query.strip():
        mask &= view[ID_COL].str.contains(query.strip(), case=False, regex=False)
    result = view[mask].copy()

    sort_cols = [ID_COL] + NUMERIC_FEATURES + ["Performance"] + (["Predicted_Performance"] if has_pred else [])
    s1, s2 = st.columns([2, 1])
    sort_col = s1.selectbox("Sort by", sort_cols, index=1)
    ascending = s2.radio("Order", ["Ascending", "Descending"], horizontal=True) == "Ascending"
    for col in ("Performance", "Predicted_Performance"):
        if col in result.columns:
            result[col] = pd.Categorical(result[col], categories=CLASS_ORDER, ordered=True)
    result = result.sort_values(sort_col, ascending=ascending, kind="mergesort")
    for col in ("Performance", "Predicted_Performance"):
        if col in result.columns:
            result[col] = result[col].astype(str)

    st.markdown(f"**{len(result)}** of {len(view)} students match the filters.")
    if result.empty:
        st.info("No students match the current filters. Try widening them.")
    else:
        st.dataframe(result, hide_index=True)
        st.download_button("Download filtered records (CSV)", csv_bytes(result), "filtered_student_records.csv", "text/csv")


def _source_tab() -> None:
    ss = st.session_state
    st.markdown(f"**Active dataset:** {ss.source_name}")
    if ss.is_synthetic:
        st.caption("Synthetic data – randomly generated for demonstration; not real student records.")

    section("1. Use the built-in sample dataset")
    n_records = st.number_input("Number of synthetic records", 200, 2000, 300, 50)
    if st.button("Load synthetic sample dataset"):
        load_sample(int(n_records))
        st.success(f"Loaded {int(n_records)} synthetic records and retrained the model.")
        st.rerun()

    section("2. Upload your own CSV")
    st.caption(f"Required columns: {', '.join(ALL_FEATURES)}, {TARGET}. Optional: {ID_COL}. {TARGET} must contain Low, Medium or High.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="dataset_upload")
    if uploaded is not None:
        try:
            raw = read_csv_upload(uploaded)
        except DatasetError as exc:
            st.error(str(exc))
        else:
            st.markdown(f"Uploaded file **{uploaded.name}** – {raw.shape[0]} rows × {raw.shape[1]} columns")
            st.dataframe(raw.head(10), hide_index=True)
            remove_dups = st.checkbox("Remove duplicate records", value=True)
            if st.button("Process dataset and train model", type="primary"):
                try:
                    df, report = clean_dataset(raw, remove_dups)
                except DatasetError as exc:
                    st.error(str(exc))
                else:
                    set_dataset(df, report, f"Uploaded: {uploaded.name}", False)
                    if ss.model is None:
                        st.warning(f"Dataset loaded, but the model could not be trained: {ss.train_error}")
                    else:
                        st.success("Dataset processed and model retrained. Check the Dashboard and Model Evaluation pages.")
                    st.rerun()

    section("3. Downloads")
    d1, d2 = st.columns(2)
    d1.download_button("Download CSV template", csv_bytes(template_dataframe()), "student_dataset_template.csv", "text/csv")
    d2.download_button("Download current dataset", csv_bytes(ss.df), "current_dataset.csv", "text/csv")


def page_dataset() -> None:
    page_header("Dataset Management", "Upload, inspect, clean and search the student dataset.")
    tab_src, tab_over, tab_rec = st.tabs(["Data Source", "Overview & Quality", "Student Records"])
    with tab_src:
        _source_tab()
    with tab_over:
        _overview_tab()
    with tab_rec:
        _records_tab()


def page_training() -> None:
    ss = st.session_state
    page_header("Model Training", "Configure the Decision Tree and retrain it on the current dataset.")
    st.markdown(
        '<div class="note-box">Algorithm: <b>DecisionTreeClassifier(criterion="entropy", random_state=42)</b> &nbsp;|&nbsp; '
        'Split: <b>80% train / 20% test</b> (stratified when possible) &nbsp;|&nbsp; Preprocessing is fitted on the training split only.</div>',
        unsafe_allow_html=True,
    )
    params = ss.params
    section("Hyperparameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        limit = st.checkbox("Limit maximum tree depth", value=params["limit_depth"])
        depth = st.slider("Maximum tree depth", 1, 20, int(params["max_depth"]), disabled=not limit)
    with c2:
        min_split = st.slider("Minimum samples to split a node", 2, 50, int(params["min_samples_split"]))
    with c3:
        min_leaf = st.slider("Minimum samples per leaf", 1, 50, int(params["min_samples_leaf"]))

    section("Feature Selection")
    selected = st.multiselect("Features used by the model", ALL_FEATURES, default=ss.features, format_func=lambda f: LABELS[f])

    if st.button("Train / Retrain Model", type="primary"):
        if not selected:
            st.error("Select at least one feature.")
        else:
            ss.params = {"limit_depth": limit, "max_depth": depth, "min_samples_split": min_split, "min_samples_leaf": min_leaf}
            ss.features = list(selected)
            with st.spinner("Training Decision Tree..."):
                retrain()
            if ss.model is None:
                st.error(ss.train_error)
            else:
                st.success("Model trained successfully.")

    if ss.model is None:
        st.error(ss.train_error or "No model has been trained yet.")
        return
    b = ss.model
    section("Training Summary")
    metric_row(
        [
            metric_card("Training samples", f"{b['n_train']}", "80% of the data"),
            metric_card("Test samples", f"{b['n_test']}", "Held out, never seen in training"),
            metric_card("Train accuracy", f"{b['train_accuracy']:.1%}"),
            metric_card("Test accuracy", f"{b['metrics']['accuracy']:.1%}"),
        ]
    )
    metric_row(
        [
            metric_card("Tree depth", f"{b['depth']}"),
            metric_card("Leaf nodes", f"{b['n_leaves']}"),
            metric_card("Features used", f"{len(b['features'])}"),
            metric_card("Stratified split", "Yes" if b["stratified"] else "No", "Class ratios preserved" if b["stratified"] else "Not enough samples per class"),
        ]
    )
    gap = b["train_accuracy"] - b["metrics"]["accuracy"]
    if gap > 0.15:
        st.warning(f"Train accuracy is {gap:.0%} higher than test accuracy – the tree may be overfitting. Try a smaller maximum depth or larger minimum-samples values.")
    else:
        st.success("Train and test accuracy are reasonably close, so there is no strong sign of overfitting.")
    p = b["params"]
    st.caption(
        f"Parameters used: max_depth={p['max_depth'] if p['limit_depth'] else 'None (unlimited)'}, "
        f"min_samples_split={p['min_samples_split']}, min_samples_leaf={p['min_samples_leaf']}, criterion=entropy, random_state=42"
    )


def page_evaluation() -> None:
    page_header("Model Evaluation", "All metrics are calculated on the held-out test set (20% of the data).")
    if not require_model():
        return
    b = st.session_state.model
    m = b["metrics"]
    metric_row(
        [
            metric_card("Accuracy", f"{m['accuracy']:.1%}"),
            metric_card("Precision", f"{m['precision']:.1%}", "Weighted average"),
            metric_card("Recall", f"{m['recall']:.1%}", "Weighted average"),
            metric_card("F1-score", f"{m['f1']:.1%}", "Weighted average"),
        ]
    )
    if b["missing_test_classes"]:
        st.warning("The test split contains no students from: " + ", ".join(b["missing_test_classes"]) + ". Their metrics show 0 with support 0.")
    if not b["stratified"]:
        st.info("The split could not be stratified because some classes have very few samples; class ratios in the test set may differ.")

    left, right = st.columns(2)
    with left:
        section("Confusion Matrix")
        normalise = st.checkbox("Show row percentages instead of counts", value=False)
        show_fig(fig_confusion(b["confusion"], normalise))
    with right:
        section("Classification Report")
        st.dataframe(b["report_df"].round(3))
        st.caption("Support = number of test students in each class.")
        section("Feature Importance")
        show_fig(fig_importance(b["importances"]))
    st.caption("Feature importance shows which features the tree used most to reduce entropy. It describes the model, not cause and effect.")
    if b["n_test"] < 100:
        st.caption(f"Note: the test set has only {b['n_test']} students, so these metrics can change noticeably with a different split or dataset.")


def page_tree() -> None:
    page_header("Decision Tree Visualization", "Inspect the trained tree: each box shows the split rule, entropy, samples and predicted class.")
    if not require_model():
        return
    b = st.session_state.model
    max_depth = max(b["depth"], 1)
    display_depth = st.slider("Levels to display", 1, max_depth, min(3, max_depth))
    if display_depth > 5:
        st.warning("Showing many levels makes the picture very large; zoom in or download it for a closer look.")
    st.caption("Class Participation is encoded as Low = 0, Medium = 1, High = 2, so a rule like 'Class Participation ≤ 0.50' means 'Low'. Colours show the majority class; darker means purer.")
    fig = fig_tree(b, display_depth)
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.download_button("Download tree image (PNG)", buf.getvalue(), "decision_tree.png", "image/png")
    with st.expander("Show the tree as text rules"):
        model = b["pipeline"].named_steps["model"]
        st.code(export_text(model, feature_names=[SHORT[f] for f in b["feature_names"]], decimals=2), language="text")


def page_batch() -> None:
    ss = st.session_state
    page_header("Batch Prediction", "Upload many students at once and download the predicted performance for each.")
    if not require_model():
        return
    b = ss.model
    st.markdown("Required columns: " + ", ".join(f"`{c}`" for c in b["features"]) + f". A `{TARGET}` column is not needed (it is ignored if present).")
    example = template_dataframe().drop(columns=[TARGET])
    st.download_button("Download batch input template", csv_bytes(example), "batch_input_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload student CSV", type=["csv"], key="batch_upload")
    if uploaded is not None:
        try:
            raw = read_csv_upload(uploaded)
        except DatasetError as exc:
            st.error(str(exc))
            return
        st.markdown(f"**{uploaded.name}** – {raw.shape[0]} rows × {raw.shape[1]} columns")
        st.dataframe(raw.head(10), hide_index=True)
        std = standardize_columns(raw)
        missing = [c for c in b["features"] if c not in std.columns]
        if missing:
            st.error("The file is missing required column(s): " + ", ".join(missing))
            return
        if st.button("Generate predictions", type="primary"):
            try:
                coerced, stats = coerce_features(std, b["features"])
                pred, proba = predict_frame(b, coerced)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
                return
            out = raw.reset_index(drop=True).copy()
            out["Predicted_Performance"] = pred
            for cls in CLASS_ORDER:
                out[f"Prob_{cls}_%"] = (proba[cls].to_numpy() * 100).round(1)
            ss.batch_result = out
            ss.batch_name = uploaded.name
            ss.batch_notes = {
                "rows_imputed": int(coerced[b["features"]].isna().any(axis=1).sum()),
                "invalid": stats["non_numeric"] + stats["out_of_range"] + stats["invalid_category"],
            }

    out = ss.batch_result
    if out is not None:
        section(f"Predictions for {ss.batch_name}")
        notes = ss.get("batch_notes", {})
        if notes.get("invalid"):
            st.warning(f"{notes['invalid']} invalid value(s) (non-numeric, out of range or unknown category) were treated as missing.")
        if notes.get("rows_imputed"):
            st.info(f"{notes['rows_imputed']} row(s) had missing values that were filled using the training data's median / most frequent value.")
        counts = out["Predicted_Performance"].value_counts()
        metric_row([metric_card(f"Predicted {c}", f"{counts.get(c, 0)}", f"{counts.get(c, 0) / len(out):.1%}", CLASS_COLORS[c]) for c in CLASS_ORDER])
        st.dataframe(out, hide_index=True)
        stem = re.sub(r"\.csv$", "", ss.batch_name, flags=re.I) or "batch"
        st.download_button("Download predictions (CSV)", csv_bytes(out), f"predictions_{stem}.csv", "text/csv")


def page_about() -> None:
    page_header("About Project", "Student Performance Prediction Using Decision Tree Algorithm")
    section("Project Objective")
    st.write(
        "To build a Machine Learning application that predicts a student's academic performance category "
        "(Low, Medium or High) from attendance, examination marks, assignment scores, study habits and class participation."
    )
    section("Problem Statement")
    st.write(
        "Teachers often notice struggling students only after final results are published. An early, data-driven indicator "
        "helps colleges identify students who may need extra support while there is still time to act."
    )
    section("Input Features")
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": [LABELS[f] for f in ALL_FEATURES],
                "Type": ["Number"] * 4 + ["Number (hours/day)", "Category", "Number"][0:0] + [],
            }
        ).iloc[0:0]
        if False
        else pd.DataFrame(
            {
                "Feature": [LABELS[f] for f in ALL_FEATURES],
                "Type": ["Numeric", "Numeric", "Numeric", "Numeric", "Numeric", "Numeric", "Categorical (Low / Medium / High)"],
                "Valid range": ["0–100", "0–100", "0–100", "0–100", "0–24", "0–100", "Low, Medium, High"],
            }
        ),
        hide_index=True,
    )
    st.write("Target: **Performance** – Low, Medium or High.")
    section("Machine Learning Algorithm: Decision Tree Classification")
    st.write(
        "A Decision Tree is a flowchart-like model. Each internal node asks a question about one feature "
        "(for example *'Is Internal Marks ≤ 62.5?'*), each branch is an answer, and each leaf gives a predicted class. "
        "To predict, a student simply follows the questions from the top (root) to a leaf."
    )
    section("How Entropy and Information Gain Work")
    st.write(
        "**Entropy** measures how mixed a group of students is. A group with only High performers has entropy 0 (pure); "
        "an even mix of Low, Medium and High has the highest entropy."
    )
    st.latex(r"H(S) = -\sum_{i} p_i \log_2 p_i")
    st.write(
        "**Information Gain** is how much a question reduces entropy. At every node the algorithm tries many possible questions "
        "and picks the one with the highest gain, then repeats on each branch until the groups are pure or a stopping rule "
        "(maximum depth, minimum samples) is reached."
    )
    st.latex(r"IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)")
    section("Technologies Used")
    st.markdown(
        "- **Python** – programming language\n- **Streamlit** – web interface\n- **Scikit-learn** – Decision Tree, pipeline, metrics\n"
        "- **Pandas & NumPy** – data processing\n- **Matplotlib & Seaborn** – visualisations\n- **CSV** – dataset format"
    )
    section("Expected Outcome")
    st.write(
        "A working application that trains a Decision Tree on student data, reports honest test-set metrics, explains each "
        "prediction, and supports single and batch predictions that teachers can use as an early-warning aid."
    )
    st.info(
        "The built-in sample dataset is synthetic (randomly generated). Predictions are statistical patterns, not certainties, "
        "and feature importance does not prove cause and effect. Use results only as decision support."
    )


PAGE_FUNCTIONS = {
    "Dashboard": page_dashboard,
    "Individual Student Prediction": page_prediction,
    "Dataset Management": page_dataset,
    "Model Training": page_training,
    "Model Evaluation": page_evaluation,
    "Decision Tree Visualization": page_tree,
    "Batch Prediction": page_batch,
    "About Project": page_about,
}


def main() -> None:
    st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="wide")
    render_html(CSS)
    init_state()
    render_sidebar()
    PAGE_FUNCTIONS[st.session_state.get("nav", PAGES[0])]()


main()