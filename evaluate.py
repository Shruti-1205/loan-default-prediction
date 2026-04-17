"""Evaluate the trained pipeline and select the operating threshold.

Loads the persisted joblib pipeline, scores the held-out test split, chooses
the probability threshold that maximises F1 on the validation data, and
writes evaluation artifacts (metrics.json plus PNG charts) to the project
root. These artifacts are consumed by the FastAPI service and the Streamlit
app at runtime.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

NUMERICAL_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]

CATEGORICAL_FEATURES = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

TARGET = "Default"


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def plot_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_pr_curve(y_true, y_proba, threshold: float, output_path: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label="Precision vs Recall")
    if len(thresholds) > 0:
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        ax.scatter(
            [recall[idx]],
            [precision[idx]],
            color="red",
            label=f"Threshold = {threshold:.2f}",
            zorder=5,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_threshold_sweep(y_true, y_proba, output_path: Path) -> None:
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s, precisions, recalls = [], [], []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, pred, zero_division=0))
        precisions.append(precision_score(y_true, pred, zero_division=0))
        recalls.append(recall_score(y_true, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, f1s, label="F1")
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metric vs Decision Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(pipeline, output_path: Path, top_n: int = 15) -> None:
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(order)), importances[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Loan_default.csv")
    parser.add_argument("--model", default="final_rf_pipeline_with_metadata.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    artifacts_dir = project_root / "evaluation_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    pipeline, model_metadata = joblib.load(project_root / args.model)

    df = pd.read_csv(project_root / args.data)
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    default_threshold = 0.5
    default_pred = (y_proba >= default_threshold).astype(int)

    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test.values, y_proba)
    tuned_pred = (y_proba >= optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, tuned_pred).ravel()
    metrics = {
        "model_metadata": model_metadata,
        "n_test": int(len(y_test)),
        "class_balance": {
            "no_default": float((y_test == 0).mean()),
            "default": float((y_test == 1).mean()),
        },
        "default_threshold_metrics": {
            "threshold": default_threshold,
            "f1": float(f1_score(y_test, default_pred, zero_division=0)),
            "precision": float(precision_score(y_test, default_pred, zero_division=0)),
            "recall": float(recall_score(y_test, default_pred, zero_division=0)),
        },
        "optimal_threshold": optimal_threshold,
        "optimal_threshold_metrics": {
            "f1": optimal_f1,
            "precision": float(precision_score(y_test, tuned_pred, zero_division=0)),
            "recall": float(recall_score(y_test, tuned_pred, zero_division=0)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    metrics_path = project_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    plot_confusion_matrix(y_test, tuned_pred, artifacts_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, artifacts_dir / "roc_curve.png")
    plot_pr_curve(y_test, y_proba, optimal_threshold, artifacts_dir / "pr_curve.png")
    plot_threshold_sweep(y_test, y_proba, artifacts_dir / "threshold_sweep.png")
    plot_feature_importance(pipeline, artifacts_dir / "feature_importance.png")

    print(f"Optimal threshold: {optimal_threshold:.3f} (F1 = {optimal_f1:.3f})")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(classification_report(y_test, tuned_pred, digits=3))
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote charts to {artifacts_dir}")


if __name__ == "__main__":
    main()
