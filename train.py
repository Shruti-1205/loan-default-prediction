"""Train the loan default classifier and persist a scikit-learn Pipeline.

This is a clean, reproducible alternative to generate_model.py. It trains the
Random Forest model used in production and saves the fitted pipeline along
with metadata to a joblib file.

Usage:
    python train.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def build_pipeline(random_state: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET]
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Loan_default.csv", help="Training CSV path")
    parser.add_argument(
        "--output",
        default="final_rf_pipeline_with_metadata.joblib",
        help="Output joblib path",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_path = project_root / args.data
    output_path = project_root / args.output

    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(random_state=args.random_state)
    pipeline.fit(X_train, y_train)

    metadata = {
        "scikit_learn_version": sklearn.__version__,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "random_state": args.random_state,
    }

    joblib.dump((pipeline, metadata), output_path)
    print(f"Saved model to {output_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
