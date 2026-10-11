"""
Heart Disease Prediction - Student Demo App
-------------------------------------------
A small Streamlit web app that loads the Support Vector Classifier trained in
the Heart Disease notebook (heart_disease_svc_model.pkl) and recreates the same
LabelEncoders (fit on the same fixed category lists), so predictions match the
notebook exactly.

Educational demo only - not a medical tool.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_disease_svc_model.pkl"

SEX = ["Male", "Female"]
CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
FASTING_BS = ["Normal", "High"]
EXERCISE_ANGINA = ["No", "Yes"]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """Same fixed category lists as the notebook, so the codes are identical."""
    return {
        "Sex": LabelEncoder().fit(SEX),
        "ChestPain": LabelEncoder().fit(CHEST_PAIN),
        "FastingBS": LabelEncoder().fit(FASTING_BS),
        "ExerciseAngina": LabelEncoder().fit(EXERCISE_ANGINA),
    }


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Enter a patient's details below and the trained Support Vector Classifier (SVC)
    will predict whether heart disease is likely. Built from the UCI Heart Disease
    (Cleveland) dataset using 4 input attributes.
    """
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", SEX)
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN, index=3)
with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar (> 120 mg/dl)", FASTING_BS)
    exercise_angina = st.selectbox("Exercise-Induced Angina", EXERCISE_ANGINA)

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "FastingBS": [fasting_bs],
        "ExerciseAngina": [exercise_angina],
    })

    encoded = pd.DataFrame({
        col: encoders[col].transform(user_input[col]) for col in user_input.columns
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ NO HEART DISEASE LIKELY")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Educational demo built from the UCI Heart Disease (Cleveland) dataset. "
    "It is not a medical tool and must not be used for diagnosis."
)
