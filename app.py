import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

model = load_model()

with st.sidebar:

    st.header("⚙️ Settings")

    show_probability = st.checkbox(
        "Show Probability Chart",
        value=True
    )

    show_raw = st.checkbox(
        "Show Raw Input",
        value=False
    )

    st.markdown("---")

    st.markdown("## About")

    st.write("""
This application predicts the likelihood of heart disease using a
Gradient Boosting Machine trained on 10,000 patient records.

Model:
- Gradient Boosting
- GridSearchCV Optimized

Features:
- Lifestyle
- Blood Pressure
- Cholesterol
- Biomarkers
- Family History
""")

st.title("❤️ Heart Disease Risk Predictor")

st.write(
    "Enter the patient's clinical and lifestyle information below."
)

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:

    age = st.slider(
        "Age",
        18,
        100,
        45
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    bp = st.number_input(
        "Blood Pressure",
        70,
        250,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        100,
        500,
        200
    )
    with col2:
        bmi = st.number_input(
            "BMI",
            10.0,
            60.0,
            25.0
        )

        exercise = st.selectbox(
            "Exercise Habits",
            ["Low", "Medium", "High"]
        )

        smoking = st.selectbox(
            "Smoking",
            ["No", "Yes"]
        )

        alcohol = st.selectbox(
            "Alcohol Consumption",
            ["Low", "Medium", "High"]
        )

with col3:

    diabetes = st.selectbox(
        "Diabetes",
        ["No","Yes"]
    )

    family = st.selectbox(
        "Family Heart Disease",
        ["No","Yes"]
    )

    stress = st.selectbox(
        "Stress Level",
        ["Low","Medium","High"]
    )

    sleep = st.slider(
        "Sleep Hours",
        1,
        12,
        7
    )

low_hdl = st.selectbox(
    "Low HDL Cholesterol",
    ["No","Yes"]
)

high_ldl = st.selectbox(
    "High LDL Cholesterol",
    ["No","Yes"]
)

high_bp = st.selectbox(
    "High Blood Pressure",
    ["No","Yes"]
)

sugar = st.selectbox(
    "Sugar Consumption",
    ["Low","Medium","High"]
)

triglycerides = st.number_input(
    "Triglyceride Level",
    20.0,
    600.0,
    150.0
)

fasting = st.number_input(
    "Fasting Blood Sugar",
    50.0,
    300.0,
    100.0
)

crp = st.number_input(
    "CRP Level",
    0.0,
    20.0,
    3.0
)

homocysteine = st.number_input(
    "Homocysteine Level",
    0.0,
    100.0,
    10.0
)

st.markdown("---")

if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):

    # -------------------------------
    # Feature Engineering
    # -------------------------------

    age_bp = age * bp

    chol_bmi = cholesterol * bmi

    inflammation_score = crp + homocysteine

    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    lifestyle_risk = (
        (smoking == "Yes") +
        (alcohol == "High") +
        (exercise == "Low")
    )

    cardio_risk = (
        (diabetes == "Yes") +
        (high_bp == "Yes") +
        (high_ldl == "Yes") +
        (low_hdl == "Yes")
    )

    family_risk = (
        (family == "Yes") +
        (diabetes == "Yes")
    )

    # -------------------------------
    # Create DataFrame
    # -------------------------------

    input_data = pd.DataFrame({

        "Age":[age],
        "Gender":[gender],
        "Blood Pressure":[bp],
        "Cholesterol Level":[cholesterol],
        "Exercise Habits":[exercise],
        "Smoking":[smoking],
        "Family Heart Disease":[family],
        "Diabetes":[diabetes],
        "BMI":[bmi],
        "High Blood Pressure":[high_bp],
        "Low HDL Cholesterol":[low_hdl],
        "High LDL Cholesterol":[high_ldl],
        "Alcohol Consumption":[alcohol],
        "Stress Level":[stress],
        "Sleep Hours":[sleep],
        "Sugar Consumption":[sugar],
        "Triglyceride Level":[triglycerides],
        "Fasting Blood Sugar":[fasting],
        "CRP Level":[crp],
        "Homocysteine Level":[homocysteine],

        "BMI_Category":[bmi_category],
        "Age_BP":[age_bp],
        "Chol_BMI":[chol_bmi],
        "Inflammation_Score":[inflammation_score],
        "Lifestyle_Risk":[lifestyle_risk],
        "Cardio_Risk":[cardio_risk],
        "Family_Risk":[family_risk]

    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.markdown("## Prediction Result")

    left, right = st.columns(2)

    with left:

        if prediction == 1:

            st.error("❤️ High Risk of Heart Disease")

            st.metric(
                "Heart Disease Probability",
                f"{probability[1]*100:.1f}%"
            )

        else:

            st.success("💚 Low Risk of Heart Disease")

            st.metric(
                "Healthy Heart Probability",
                f"{probability[0]*100:.1f}%"
            )

    with right:

        if show_probability:

            chart = pd.DataFrame(
                {
                    "Probability":[
                        probability[0]*100,
                        probability[1]*100
                    ]
                },
                index=[
                    "Low Risk",
                    "High Risk"
                ]
            )

            st.bar_chart(chart)

    if show_raw:

        st.subheader("Input Data")

        st.dataframe(input_data)

    st.info(
        """
        **Disclaimer**

        This application is intended solely for educational and demonstration
        purposes.

        It should not be used as a substitute for professional medical
        diagnosis, treatment, or advice.

        Always consult a qualified healthcare professional regarding any
        medical concerns.
        """
    )

st.markdown("---")

st.caption(
    "Built with Streamlit | Heart Disease Prediction using Gradient Boosting | Data Science & AI Portfolio Project"
)

