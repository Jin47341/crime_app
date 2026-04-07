import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Chicago Crime Risk Prediction", layout="centered")

st.title("Chicago Crime Risk Prediction")
st.write("Enter the weekly crime features below to predict whether the next week is high risk.")

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("chicago_crime_model_xgboost_tuned.joblib")
    return model

model = load_model()

# ----------------------------
# Input fields
# ----------------------------
community_area = st.number_input("Community Area", min_value=1, max_value=77, value=25, step=1)
lag_crime_1 = st.number_input("Lag Crime 1 (last week crime count)", min_value=0.0, value=50.0, step=1.0)
lag_crime_2 = st.number_input("Lag Crime 2 (two weeks ago)", min_value=0.0, value=48.0, step=1.0)
lag_crime_4 = st.number_input("Lag Crime 4 (four weeks ago)", min_value=0.0, value=52.0, step=1.0)
rolling_mean_4 = st.number_input("Rolling Mean 4", min_value=0.0, value=49.0, step=0.1)
month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
week_of_year = st.number_input("Week of Year", min_value=1, max_value=53, value=24, step=1)

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Community Area": community_area,
        "lag_crime_1": lag_crime_1,
        "lag_crime_2": lag_crime_2,
        "lag_crime_4": lag_crime_4,
        "rolling_mean_4": rolling_mean_4,
        "month": month,
        "week_of_year": week_of_year
    }])

    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"High Risk Week Predicted")
        else:
            st.success(f"Low Risk Week Predicted")

        st.write(f"Risk probability: **{proba:.4f}**")

    except Exception as e:
        st.error("Prediction failed. Please check whether the model and input feature names match.")
        st.code(str(e))