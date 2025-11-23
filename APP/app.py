import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    with open("../model/symptoms.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("🧠 Alzheimer Diagnosis Prediction App")
st.write("This app predicts the probability of Alzheimer's diagnosis using a trained Gradient Boosting model.")

st.sidebar.header("Patient Input Features")

def user_input():
    MMSE = st.sidebar.text_input("MMSE Score (0-30)", "20")
    FunctionalAssessment = st.sidebar.text_input("Functional Assessment Score (0-10)", "5")
    MemoryComplaints = st.sidebar.text_input("Memory Complaints Score (0-10)", "4")
    BehavioralProblems = st.sidebar.text_input("Behavioral Problems Score (0-10)", "3")
    ADL = st.sidebar.text_input("ADL Score (0-10)", "6")

    data = {
        'MMSE': float(MMSE),
        'FunctionalAssessment': float(FunctionalAssessment),
        'MemoryComplaints': float(MemoryComplaints),
        'BehavioralProblems': float(BehavioralProblems),
        'ADL': float(ADL)
    }

    return pd.DataFrame([data])

input_df = user_input()

st.subheader(" Patient Data")
st.write(input_df)

if st.button(" Predict Alzheimer's Diagnosis"):

    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    st.subheader(" Prediction Result:")
    if prediction == 1:
        st.error(" Model Prediction: Patient is likely to have Alzheimer's Disease.")
    else:
        st.success(" Model Prediction: Patient is unlikely to have Alzheimer's Disease.")

    st.subheader(" Prediction Probability")
    st.write(f"Probability of Diagnosis = {prediction_prob:.2f}")
