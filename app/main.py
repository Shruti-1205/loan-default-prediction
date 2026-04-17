"""FastAPI service for loan default prediction.

The endpoint returns the class prediction, the probability of default, the
threshold that was applied, and the top SHAP feature contributions for the
request. The threshold is loaded from metrics.json (produced by
evaluate.py); if the file is missing the service falls back to 0.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "final_rf_pipeline_with_metadata.joblib"
METRICS_PATH = PROJECT_ROOT / "metrics.json"


def load_threshold() -> float:
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        return float(metrics.get("optimal_threshold", 0.5))
    return 0.5


pipeline, metadata = joblib.load(MODEL_PATH)
threshold = load_threshold()
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]
feature_names = list(preprocessor.get_feature_names_out())
explainer = shap.TreeExplainer(classifier)

app = FastAPI(
    title="Loan Default Prediction API",
    description="Random Forest classifier with SHAP explanations and a tuned decision threshold.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Income: float = Field(..., ge=0)
    LoanAmount: float = Field(..., ge=0)
    CreditScore: float = Field(..., ge=300, le=850)
    MonthsEmployed: int = Field(..., ge=0)
    NumCreditLines: int = Field(..., ge=0)
    InterestRate: float = Field(..., ge=0)
    LoanTerm: int = Field(..., ge=1)
    DTIRatio: float = Field(..., ge=0, le=1)
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class PredictionResponse(BaseModel):
    prediction: int
    probability_of_default: float
    threshold_used: float
    risk_label: str
    top_feature_contributions: list[FeatureContribution]


def _class_one_shap(shap_output, n_features: int) -> np.ndarray:
    if isinstance(shap_output, list):
        return np.asarray(shap_output[1])
    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


@app.get("/")
async def read_root() -> dict:
    return {
        "service": "Loan Default Prediction API",
        "version": "1.0.0",
        "threshold": threshold,
        "model": metadata,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {"detail": "metrics.json not found. Run evaluate.py to generate it."}
    return json.loads(METRICS_PATH.read_text())


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    input_df = pd.DataFrame([request.dict()])
    probability = float(pipeline.predict_proba(input_df)[0, 1])
    prediction = int(probability >= threshold)
    risk_label = "High Risk of Default" if prediction == 1 else "Low Risk of Default"

    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    shap_row = _class_one_shap(explainer.shap_values(transformed), len(feature_names))[0]
    values_row = np.asarray(transformed)[0]

    ranked = sorted(
        zip(feature_names, values_row, shap_row),
        key=lambda item: abs(item[2]),
        reverse=True,
    )[:10]

    contributions = [
        FeatureContribution(feature=name, value=float(val), shap_value=float(sv))
        for name, val, sv in ranked
    ]

    return PredictionResponse(
        prediction=prediction,
        probability_of_default=probability,
        threshold_used=threshold,
        risk_label=risk_label,
        top_feature_contributions=contributions,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
