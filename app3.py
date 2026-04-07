from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Chicago Crime Risk Dashboard",
    layout="wide",
)

MODEL_PATH = Path("chicago_crime_model_xgboost_tuned.joblib")

FEATURE_COLUMNS = [
    "Community Area",
    "lag_crime_1",
    "lag_crime_2",
    "lag_crime_4",
    "rolling_mean_4",
    "month",
    "week_of_year",
]

DEFAULT_INPUTS = {
    "Community Area": 25,
    "lag_crime_1": 50.0,
    "lag_crime_2": 48.0,
    "lag_crime_4": 52.0,
    "rolling_mean_4": 49.0,
    "month": 6,
    "week_of_year": 24,
}

EXAMPLE_SCENARIOS = {
    "high_risk": {
        "Community Area": 25,
        "lag_crime_1": 88.0,
        "lag_crime_2": 81.0,
        "lag_crime_4": 79.0,
        "rolling_mean_4": 82.0,
        "month": 8,
        "week_of_year": 33,
    },
    "low_risk": {
        "Community Area": 47,
        "lag_crime_1": 18.0,
        "lag_crime_2": 20.0,
        "lag_crime_4": 17.0,
        "rolling_mean_4": 18.5,
        "month": 2,
        "week_of_year": 7,
    },
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1rem;
        }
        section[data-testid="stSidebar"] .stButton > button {
            padding-top: 0.45rem;
            padding-bottom: 0.45rem;
        }
        .hero-card,
        .result-card,
        .disclaimer-card {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96));
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.06);
            color: #0f172a;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.85rem;
        }
        .hero-title {
            font-size: 2.05rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #334155;
            line-height: 1.55;
            margin-bottom: 0;
        }
        .status-high {
            border-left: 6px solid #b91c1c;
            background: #fef2f2;
            color: #7f1d1d;
        }
        .status-low {
            border-left: 6px solid #15803d;
            background: #f0fdf4;
            color: #14532d;
        }
        .status-high h4,
        .status-high .status-copy,
        .status-high .status-note {
            color: #7f1d1d !important;
        }
        .status-low h4,
        .status-low .status-copy,
        .status-low .status-note {
            color: #14532d !important;
        }
        .muted-text {
            color: #475569;
            font-size: 0.95rem;
        }
        .small-note {
            color: #64748b;
            font-size: 0.86rem;
        }
        .disclaimer-card {
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    for key, value in DEFAULT_INPUTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_scenario(values: dict) -> None:
    for key, value in values.items():
        st.session_state[key] = value


def reset_inputs() -> None:
    apply_scenario(DEFAULT_INPUTS)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_input_dataframe(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([{column: inputs[column] for column in FEATURE_COLUMNS}])[FEATURE_COLUMNS]


def get_probability_scores(model, input_df: pd.DataFrame) -> tuple[float, float]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        if len(probabilities) >= 2:
            return float(probabilities[0]), float(probabilities[1])

    prediction = int(model.predict(input_df)[0])
    return (0.0, 1.0) if prediction == 1 else (1.0, 0.0)


def predict_single(model, inputs: dict) -> dict:
    input_df = build_input_dataframe(inputs)
    prediction = int(model.predict(input_df)[0])
    prob_low, prob_high = get_probability_scores(model, input_df)

    return {
        "prediction": prediction,
        "risk_probability": prob_high,
        "low_probability": prob_low,
    }


def interpret_probability(prediction: int, probability: float) -> tuple[str, str]:
    if prediction == 1:
        if probability >= 0.80:
            return "High-risk week predicted", "The current feature pattern shows a strong high-risk signal."
        if probability >= 0.60:
            return "Elevated risk week predicted", "The model detects a moderate-to-strong high-risk signal."
        return "Borderline high-risk week predicted", "The model predicts high risk, but the signal is relatively weak."

    if probability <= 0.20:
        return "Low-risk week predicted", "The submitted pattern is less consistent with prior higher-risk weeks."
    if probability <= 0.40:
        return "Lower-risk week predicted", "The model does not currently classify the week as high risk."
    return "Low-risk classification with some uncertainty", "The model predicts low risk, but the case is close to the decision boundary."


def validate_inputs(inputs: dict) -> list[str]:
    warnings = []

    if inputs["lag_crime_1"] > 300 or inputs["lag_crime_2"] > 300 or inputs["lag_crime_4"] > 300:
        warnings.append("One or more weekly crime counts are unusually high. Please double-check the values.")

    if abs(inputs["rolling_mean_4"] - inputs["lag_crime_1"]) > 100:
        warnings.append("The rolling mean differs substantially from last week's count. Please confirm the input values.")

    if inputs["month"] == 2 and inputs["week_of_year"] > 12:
        warnings.append("The month and week-of-year combination looks uncommon. Please verify the calendar values.")

    if inputs["month"] in {11, 12} and inputs["week_of_year"] < 10:
        warnings.append("The month and week-of-year combination looks uncommon. Please verify the calendar values.")

    return warnings


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Academic POC / MVP</div>
            <div class="hero-title">Chicago Community Area Crime Risk Prediction</div>
            <p class="hero-subtitle">
                This app predicts whether the upcoming week in a Chicago community area is likely to be a high-risk crime week.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_available: bool) -> dict:
    st.sidebar.header("Prediction Inputs")
    st.sidebar.caption("Enter weekly crime-related features for one community area.")

    demo_col1, demo_col2 = st.sidebar.columns(2)
    if demo_col1.button("Example: High Risk", use_container_width=True):
        apply_scenario(EXAMPLE_SCENARIOS["high_risk"])
    if demo_col2.button("Example: Low Risk", use_container_width=True):
        apply_scenario(EXAMPLE_SCENARIOS["low_risk"])

    if st.sidebar.button("Reset to Default Values", use_container_width=True):
        reset_inputs()

    st.sidebar.markdown("**Weekly Feature Inputs**")

    pair_col1, pair_col2 = st.sidebar.columns(2)
    lag_col1, lag_col2 = st.sidebar.columns(2)
    history_col1, time_col1 = st.sidebar.columns(2)

    inputs = {
        "Community Area": pair_col1.number_input(
            "Community Area",
            min_value=1,
            max_value=77,
            step=1,
            key="Community Area",
            help="Chicago community area identifier from 1 to 77.",
        ),
        "lag_crime_1": pair_col2.number_input(
            "Weekly Crime Count (Last Week)",
            min_value=0.0,
            step=1.0,
            key="lag_crime_1",
        ),
        "lag_crime_2": lag_col1.number_input(
            "Weekly Crime Count (Two Weeks Ago)",
            min_value=0.0,
            step=1.0,
            key="lag_crime_2",
        ),
        "lag_crime_4": lag_col2.number_input(
            "Weekly Crime Count (Four Weeks Ago)",
            min_value=0.0,
            step=1.0,
            key="lag_crime_4",
        ),
        "rolling_mean_4": history_col1.number_input(
            "Four-Week Rolling Average",
            min_value=0.0,
            step=0.1,
            key="rolling_mean_4",
        ),
        "month": time_col1.number_input(
            "Month",
            min_value=1,
            max_value=12,
            step=1,
            key="month",
        ),
        "week_of_year": st.sidebar.number_input(
            "Week of Year",
            min_value=1,
            max_value=53,
            step=1,
            key="week_of_year",
        ),
    }

    validation_messages = validate_inputs(inputs)
    if validation_messages:
        with st.sidebar.expander("Input Checks", expanded=False):
            for message in validation_messages:
                st.warning(message)

    if not model_available:
        st.sidebar.error("The model is unavailable. Prediction is currently disabled.")

    st.sidebar.divider()
    predict_clicked = st.sidebar.button(
        "Run Prediction",
        type="primary",
        use_container_width=True,
        disabled=not model_available,
    )

    return {"inputs": inputs, "predict_clicked": predict_clicked}


def render_result_panel(result: dict | None, model_error: str | None) -> None:
    st.subheader("Prediction Result")

    if model_error:
        st.error("The model could not be loaded in the current environment.")
        with st.expander("Technical Details"):
            st.code(model_error)
        return

    if result is None:
        st.info("Enter the feature values in the sidebar and click `Run Prediction`.")
        return

    prediction = result["prediction"]
    probability = result["risk_probability"]
    title, interpretation = interpret_probability(prediction, probability)
    status_class = "status-high" if prediction == 1 else "status-low"
    status_label = "High Risk" if prediction == 1 else "Low Risk"

    top_col1, top_col2, top_col3 = st.columns(3)
    top_col1.metric("Predicted Class", status_label)
    top_col2.metric("High-Risk Probability", f"{probability:.1%}")
    top_col3.metric("Low-Risk Probability", f"{result['low_probability']:.1%}")

    st.markdown(
        f"""
        <div class="result-card {status_class}">
            <h4 style="margin-bottom:0.35rem;">{title}</h4>
            <p class="status-copy" style="margin-bottom:0.45rem;">{interpretation}</p>
            <p class="status-note small-note" style="margin-bottom:0;">
                This result is generated from the submitted feature values and the trained XGBoost model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(max(float(probability), 0.0), 1.0))
    probability_chart_df = pd.DataFrame(
        {
            "Class": ["Low Risk", "High Risk"],
            "Probability": [result["low_probability"], probability],
        }
    ).set_index("Class")
    st.bar_chart(probability_chart_df)


def render_input_summary(inputs: dict) -> None:
    st.subheader("Input Summary")

    summary_df = pd.DataFrame(
        [{"Feature": key, "Value": value} for key, value in inputs.items()]
    )

    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer-card">
            <h4 style="margin-bottom:0.4rem;">Disclaimer</h4>
            <p class="muted-text" style="margin-bottom:0.4rem;">
                This app is an academic proof-of-concept.
            </p>
            <p class="small-note" style="margin-bottom:0;">
                It is for coursework demonstration only and not for real-world policing decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    initialize_state()

    model = None
    model_error = None
    try:
        model = load_model()
    except Exception as exc:
        model_error = str(exc)

    render_header()

    sidebar_state = render_sidebar(model_available=model is not None)
    inputs = sidebar_state["inputs"]
    prediction_result = None

    if sidebar_state["predict_clicked"] and model is not None:
        try:
            prediction_result = predict_single(model, inputs)
        except Exception as exc:
            st.error("Prediction failed. Please confirm the model artifact and input feature names remain aligned.")
            with st.expander("Technical Details"):
                st.code(str(exc))

    st.divider()

    main_col, side_col = st.columns([1.7, 1.05])
    with main_col:
        render_result_panel(prediction_result, model_error)
    with side_col:
        render_input_summary(inputs)

    st.divider()
    render_disclaimer()


if __name__ == "__main__":
    main()