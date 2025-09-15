# Fake Job Posting Detector 🕵️‍♂️

A machine learning-based web application that detects fraudulent job postings using XGBoost and Logistic Regression models.
The app leverages TF-IDF text features, categorical features, and custom numeric features to accurately classify job postings as real or fake.

## Features

Predictive Models: Supports both XGBoost and Logistic Regression for binary classification.  
Text Analysis: Uses TF-IDF vectorization (unigrams + bigrams) to extract meaningful patterns from job posting text.  
Categorical Feature Handling: One-hot encoding for features like employment_type, required_experience, and required_education.  
Custom Features: Includes numeric features such as description length, suspicious keyword count, and number of exclamation marks.  
Interactive UI: Built using Streamlit for easy and intuitive web-based predictions.  
Scalable: Lightweight and deployable on Streamlit Cloud, Render, or Heroku.  

## Technologies Used

Programming Language: Python 3.11+  
Libraries & Frameworks:
    Machine Learning: scikit-learn, xgboost  
    Data Processing: numpy, pandas, scipy  
    Web App: streamlit  
    Serialization: joblib  
    Text Processing: nltk  
Version Control: Git & GitHub
