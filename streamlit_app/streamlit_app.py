"""Streamlit frontend for the loan default prediction service.

The app can run in two modes controlled by the USE_API environment variable:

- USE_API=true: calls the FastAPI service defined in app/main.py.
- USE_API=false (default): loads the joblib pipeline locally. This is the
  mode used for deployment to Streamlit Community Cloud.

Other environment variables:
    API_URL: Base URL of the FastAPI service when USE_API=true.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "final_rf_pipeline_with_metadata.joblib"
METRICS_PATH = PROJECT_ROOT / "metrics.json"
ARTIFACTS_DIR = PROJECT_ROOT / "evaluation_artifacts"
OPTIONS_PATH = APP_DIR / "streamlit_options.json"

USE_API = os.getenv("USE_API", "false").lower() == "true"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="LD",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    pipeline, metadata = joblib.load(MODEL_PATH)
    return pipeline, metadata


@st.cache_resource
def build_explainer(_pipeline):
    classifier = _pipeline.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


@st.cache_data
def load_options():
    return json.loads(OPTIONS_PATH.read_text())


def class_one_shap(shap_output) -> np.ndarray:
    if isinstance(shap_output, list):
        return np.asarray(shap_output[1])
    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def score_locally(pipeline, explainer, payload: dict, threshold: float):
    df = pd.DataFrame([payload])
    probability = float(pipeline.predict_proba(df)[0, 1])
    prediction = int(probability >= threshold)

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    transformed = preprocessor.transform(df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    shap_row = class_one_shap(explainer.shap_values(transformed))[0]
    values_row = np.asarray(transformed)[0]

    ranked = sorted(
        zip(feature_names, values_row, shap_row),
        key=lambda item: abs(item[2]),
        reverse=True,
    )[:10]

    return {
        "prediction": prediction,
        "probability_of_default": probability,
        "threshold_used": threshold,
        "risk_label": "High Risk of Default" if prediction == 1 else "Low Risk of Default",
        "top_feature_contributions": [
            {"feature": n, "value": float(v), "shap_value": float(s)} for n, v, s in ranked
        ],
    }


def score_via_api(payload: dict):
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def render_shap_chart(contributions: list[dict]) -> None:
    contributions = list(reversed(contributions))
    features = [c["feature"] for c in contributions]
    values = [c["shap_value"] for c in contributions]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(features))))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on default probability)")
    ax.set_title("Feature contributions for this applicant")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    st.image(buffer.getvalue(), use_column_width=True)


def build_sidebar_inputs(options: dict) -> dict:
    user_input = {}
    with st.sidebar:
        st.header("Applicant profile")
        st.caption("Adjust the fields and press Predict.")
        for field, (lo, hi) in options["slider_fields"].items():
            if isinstance(lo, int) and isinstance(hi, int):
                mid = (int(lo) + int(hi)) // 2
                user_input[field] = st.slider(field, int(lo), int(hi), mid)
            else:
                lo_f, hi_f = float(lo), float(hi)
                mid = round((lo_f + hi_f) / 2, 2)
                user_input[field] = st.slider(field, lo_f, hi_f, mid, step=0.01)
        for field, values in options["single_select_fields"].items():
            user_input[field] = st.selectbox(field, values)
    return user_input


def _render_prediction_block(result: dict) -> None:
    probability = result["probability_of_default"]
    prediction = result["prediction"]

    with st.container(border=True):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Probability of default", f"{probability:.2%}")
        col_b.metric("Threshold applied", f"{result['threshold_used']:.2f}")
        col_c.metric("Decision", "High risk" if prediction == 1 else "Low risk")
        if prediction == 1:
            st.error(result["risk_label"])
        else:
            st.success(result["risk_label"])
        st.progress(min(max(probability, 0.0), 1.0))

    st.markdown("### Why this prediction")
    st.caption(
        "Green bars push probability down (toward approval), "
        "red bars push it up (toward default)."
    )
    render_shap_chart(result["top_feature_contributions"])


def _render_what_if_block(result: dict) -> None:
    base = result["base"]
    alt = result["alt"]
    delta = alt["probability_of_default"] - base["probability_of_default"]
    with st.container(border=True):
        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Base probability", f"{base['probability_of_default']:.2%}")
        col_y.metric(
            "Alternate probability",
            f"{alt['probability_of_default']:.2%}",
            delta=f"{delta:+.2%}",
            delta_color="inverse",
        )
        col_z.metric(
            "Alternate decision",
            "High risk" if alt["prediction"] == 1 else "Low risk",
        )


def _apply_threshold(result: dict, threshold: float) -> dict:
    probability = result["probability_of_default"]
    prediction = int(probability >= threshold)
    return {
        **result,
        "prediction": prediction,
        "threshold_used": threshold,
        "risk_label": "High Risk of Default" if prediction == 1 else "Low Risk of Default",
    }


def render_predict_tab(pipeline, explainer, options, metrics, user_input):
    tuned_threshold = metrics.get("optimal_threshold", 0.5) if metrics else 0.5

    st.subheader("Prediction")
    st.caption(
        f"The F1 optimal threshold on the validation set is {tuned_threshold:.2f}. "
        "Drag the slider below to see how strictness changes the decision."
    )

    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(tuned_threshold),
        step=0.01,
        help="Applicants with probability of default at or above this value are flagged as high risk.",
    )

    if st.button("Predict", type="primary", use_container_width=True):
        try:
            if USE_API:
                result = score_via_api(user_input)
            else:
                result = score_locally(pipeline, explainer, user_input, threshold)
        except requests.RequestException as exc:
            st.error(f"API call failed: {exc}")
            st.session_state.pop("prediction_result", None)
            return
        st.session_state["prediction_result"] = result

    if "prediction_result" in st.session_state:
        live = _apply_threshold(st.session_state["prediction_result"], threshold)
        _render_prediction_block(live)

    st.divider()
    st.markdown("### What if analysis")
    st.caption("Compare the current applicant against an adjusted profile.")

    col1, col2, col3 = st.columns(3)
    with col1:
        alt_credit = st.slider(
            "Alternate CreditScore", 300, 850, int(user_input["CreditScore"])
        )
    with col2:
        alt_income = st.slider(
            "Alternate Income",
            options["slider_fields"]["Income"][0],
            options["slider_fields"]["Income"][1],
            int(user_input["Income"]),
        )
    with col3:
        alt_dti = st.slider(
            "Alternate DTIRatio",
            0.0,
            1.0,
            float(user_input["DTIRatio"]),
            step=0.01,
        )

    if st.button("Run what if"):
        base_result = score_locally(pipeline, explainer, user_input, threshold)
        alt_input = dict(user_input)
        alt_input["CreditScore"] = alt_credit
        alt_input["Income"] = alt_income
        alt_input["DTIRatio"] = alt_dti
        alt_result = score_locally(pipeline, explainer, alt_input, threshold)
        st.session_state["what_if_result"] = {"base": base_result, "alt": alt_result}

    if "what_if_result" in st.session_state:
        raw = st.session_state["what_if_result"]
        live = {
            "base": _apply_threshold(raw["base"], threshold),
            "alt": _apply_threshold(raw["alt"], threshold),
        }
        _render_what_if_block(live)


def show_artifact(container, filename: str, caption: str) -> None:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        container.image(
            path.read_bytes(), caption=caption, use_column_width=True
        )
    else:
        container.info(f"{filename} not found. Run python evaluate.py to generate it.")


def render_performance_tab(metrics):
    if metrics is None:
        st.warning("metrics.json not found. Run `python evaluate.py` to generate it.")
        return

    st.subheader("Model performance on held-out test set")

    optimal = metrics["optimal_threshold_metrics"]
    default = metrics["default_threshold_metrics"]

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        col2.metric("F1 (tuned)", f"{optimal['f1']:.3f}")
        col3.metric("Precision (tuned)", f"{optimal['precision']:.3f}")
        col4.metric("Recall (tuned)", f"{optimal['recall']:.3f}")
        st.caption(
            f"Optimal threshold = {metrics['optimal_threshold']:.3f}. "
            f"At the default 0.5 threshold: F1 = {default['f1']:.3f}, "
            f"precision = {default['precision']:.3f}, recall = {default['recall']:.3f}. "
            "Tuning the threshold lifts recall meaningfully on this imbalanced dataset."
        )

    st.markdown("### Diagnostic charts")
    col_a, col_b = st.columns(2)
    show_artifact(col_a, "confusion_matrix.png", "Confusion matrix")
    show_artifact(col_b, "roc_curve.png", "ROC curve")

    col_c, col_d = st.columns(2)
    show_artifact(col_c, "pr_curve.png", "Precision recall curve")
    show_artifact(col_d, "threshold_sweep.png", "Metric vs decision threshold")

    st.markdown("### Feature importance")
    show_artifact(st, "feature_importance.png", "Random Forest feature importances")


def _stack_card(container, label: str, title: str, detail: str) -> None:
    with container:
        with st.container(border=True):
            st.caption(label.upper())
            st.markdown(f"**{title}**")
            st.caption(detail)


def render_about_tab(metrics) -> None:
    st.subheader("About")

    with st.container(border=True):
        st.markdown(
            "Predicts the probability that a loan applicant will default, "
            "explains the drivers behind each prediction with SHAP, and supports "
            "what if analysis on key inputs."
        )

    if metrics is not None:
        st.markdown("### Headline performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        col2.metric(
            "F1 at tuned threshold",
            f"{metrics['optimal_threshold_metrics']['f1']:.3f}",
        )
        col3.metric("Operating threshold", f"{metrics['optimal_threshold']:.2f}")

    st.markdown("### Tech stack")
    row1_left, row1_right = st.columns(2)
    _stack_card(
        row1_left,
        "Modeling",
        "Random Forest (scikit-learn)",
        "One hot encoding, standard scaling, class balanced weights.",
    )
    _stack_card(
        row1_right,
        "Explainability",
        "SHAP TreeExplainer",
        "Per prediction feature attributions surfaced in the UI.",
    )

    row2_left, row2_right = st.columns(2)
    _stack_card(
        row2_left,
        "Evaluation",
        "Stratified holdout and threshold tuning",
        "80 / 20 split, decision threshold maximised for F1 on the validation set.",
    )
    _stack_card(
        row2_right,
        "Serving",
        "FastAPI with Pydantic",
        "Typed request validation, returns prediction, probability, and SHAP contributions.",
    )

    row3_left, row3_right = st.columns(2)
    _stack_card(
        row3_left,
        "Frontend",
        "Streamlit",
        "Prediction, explanation, and what if analysis in a single dashboard.",
    )
    _stack_card(
        row3_right,
        "Infrastructure",
        "Docker and GitHub Actions",
        "Containerised API and a CI pipeline that runs pytest on every push.",
    )


def inject_style() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1200px;}
          [data-testid="stMetricValue"] {font-size: 1.9rem;}
          [data-testid="stMetricLabel"] {opacity: 0.75;}
          h1 {margin-bottom: 0.3rem;}
          .stTabs [data-baseweb="tab-list"] {gap: 12px;}
          .stTabs [data-baseweb="tab"] {padding: 8px 14px;}
          section[data-testid="stSidebar"] h2 {margin-top: 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_style()
    st.title("Loan Default Prediction")
    st.caption(
        "Random Forest classifier with threshold tuning and SHAP explanations. "
        f"Mode: {'FastAPI backend' if USE_API else 'in-process model'}."
    )

    options = load_options()
    user_input = build_sidebar_inputs(options)
    metrics = load_metrics()

    pipeline, _ = load_pipeline()
    explainer = build_explainer(pipeline)

    tab_about, tab_predict, tab_performance = st.tabs(
        ["About", "Predict", "Model Performance"]
    )
    with tab_about:
        render_about_tab(metrics)
    with tab_predict:
        render_predict_tab(pipeline, explainer, options, metrics, user_input)
    with tab_performance:
        render_performance_tab(metrics)


if __name__ == "__main__":
    main()
else:
    main()
