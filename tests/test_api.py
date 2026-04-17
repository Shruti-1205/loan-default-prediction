"""Integration tests for the FastAPI prediction service."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "final_rf_pipeline_with_metadata.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifact is missing. Run train.py to produce it before running tests.",
)

LOW_RISK_PAYLOAD = {
    "Age": 35,
    "Income": 90000,
    "LoanAmount": 10000,
    "CreditScore": 780,
    "MonthsEmployed": 120,
    "NumCreditLines": 4,
    "InterestRate": 5.5,
    "LoanTerm": 36,
    "DTIRatio": 0.15,
    "Education": "Master's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "No",
    "HasDependents": "No",
    "LoanPurpose": "Home",
    "HasCoSigner": "Yes",
}

HIGH_RISK_PAYLOAD = {
    "Age": 22,
    "Income": 12000,
    "LoanAmount": 450000,
    "CreditScore": 320,
    "MonthsEmployed": 1,
    "NumCreditLines": 0,
    "InterestRate": 32.0,
    "LoanTerm": 360,
    "DTIRatio": 0.95,
    "Education": "High School",
    "EmploymentType": "Unemployed",
    "MaritalStatus": "Single",
    "HasMortgage": "No",
    "HasDependents": "Yes",
    "LoanPurpose": "Business",
    "HasCoSigner": "No",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def test_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "Loan Default Prediction API"
    assert "threshold" in body


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_low_risk_schema(client: TestClient) -> None:
    response = client.post("/predict", json=LOW_RISK_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert 0.0 <= body["threshold_used"] <= 1.0
    assert len(body["top_feature_contributions"]) == 10
    for item in body["top_feature_contributions"]:
        assert set(item.keys()) == {"feature", "value", "shap_value"}


def test_high_risk_probability_exceeds_low_risk(client: TestClient) -> None:
    low = client.post("/predict", json=LOW_RISK_PAYLOAD).json()
    high = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
    assert high["probability_of_default"] > low["probability_of_default"]


def test_validation_rejects_bad_input(client: TestClient) -> None:
    bad = dict(LOW_RISK_PAYLOAD)
    bad["CreditScore"] = 50
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
