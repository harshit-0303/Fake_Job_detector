import streamlit as st
import joblib
import re
import string
import numpy as np
from scipy.sparse import hstack

# Load vectorizer, encoder, models
vectorizer = joblib.load("tfidf_vectorizer.joblib")
encoder = joblib.load("onehot_encoder.joblib")
model_lr = joblib.load("logistic_regression_model.joblib")
xgb_model = joblib.load("xgboost_model.joblib")

suspicious_keywords = ["earn", "money", "income", "from home", "investment", 
                       "bonus", "credit card", "quick money", 
                       "guaranteed income", "easy cash"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

def predict_job_posting(title, company_profile, description, requirements, 
                        employment_type, required_experience, required_education,
                        model="xgb"):
    text = f"{title} {company_profile} {description} {requirements}"
    text_clean = clean_text(text)

    # Text features
    X_text = vectorizer.transform([text_clean])

    # Categorical features
    X_cat = encoder.transform([[employment_type, required_experience, required_education]])

    # Custom numeric features
    desc_length = len(text.split())
    suspicious_words = sum(1 for word in suspicious_keywords if word in text.lower())
    exclaim_count = text.count("!")
    X_custom = np.array([[desc_length, suspicious_words, exclaim_count]])

    # Final input
    X_input = hstack([X_text, X_cat, X_custom])

    if model == "lr":
        pred = model_lr.predict(X_input)[0]
    else:
        pred = xgb_model.predict(X_input)[0]

    return "🚨 Fake Job" if pred == 1 else "✅ Real Job"

# Streamlit UI
st.title("Fake Job Posting Detection 🕵️‍♂️")

title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
employment_type = st.selectbox("Employment Type", ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"])
required_experience = st.selectbox("Required Experience", ["", "Internship", "Entry level", "Mid-Senior level", "Director", "Executive"])
required_education = st.selectbox("Required Education", ["", "High School", "Associate", "Bachelor's", "Master's", "PhD", "Other"])
model_choice = st.selectbox("Choose Model", ["xgb", "lr"])

if st.button("Predict"):
    result = predict_job_posting(title, company_profile, description, requirements, 
                                 employment_type, required_experience, required_education,
                                 model_choice)
    st.subheader(f"Prediction: {result}")
