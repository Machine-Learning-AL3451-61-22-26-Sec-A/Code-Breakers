import streamlit as st
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Streamlit app title
st.title("Heart Disease Prediction using Bayesian Network")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data from CSV
    heart_disease = pd.read_csv(uploaded_file)
    st.write("The first 5 values of the dataset:")
    st.write(heart_disease.head())

    # Define the Bayesian Model
    model = BayesianModel([
        ('age', 'Lifestyle'),
        ('Gender', 'Lifestyle'),
        ('Family', 'heartdisease'),
        ('diet', 'cholestrol'),
        ('Lifestyle', 'diet'),
        ('cholestrol', 'heartdisease'),
        ('diet', 'cholestrol')
    ])

    # Fit the model using Maximum Likelihood Estimator
    model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

    # Perform inference
    HeartDisease_infer = VariableElimination(model)

    # User inputs
    st.write('### Enter the following details:')
    
    age = st.selectbox('Age', ['SuperSeniorCitizen (0)', 'SeniorCitizen (1)', 'MiddleAged (2)', 'Youth (3)', 'Teen (4)'])
    gender = st.selectbox('Gender', ['Male (0)', 'Female (1)'])
    family_history = st.selectbox('Family History', ['Yes (1)', 'No (0)'])
    diet = st.selectbox('Diet', ['High (0)', 'Medium (1)'])
    lifestyle = st.selectbox('Lifestyle', ['Athlete (0)', 'Active (1)', 'Moderate (2)', 'Sedentary (3)'])
    cholestrol = st.selectbox('Cholestrol', ['High (0)', 'BorderLine (1)', 'Normal (2)'])

    # Convert user inputs to appropriate format
    age_mapping = {'SuperSeniorCitizen (0)': 0, 'SeniorCitizen (1)': 1, 'MiddleAged (2)': 2, 'Youth (3)': 3, 'Teen (4)': 4}
    gender_mapping = {'Male (0)': 0, 'Female (1)': 1}
    family_mapping = {'Yes (1)': 1, 'No (0)': 0}
    diet_mapping = {'High (0)': 0, 'Medium (1)': 1}
    lifestyle_mapping = {'Athlete (0)': 0, 'Active (1)': 1, 'Moderate (2)': 2, 'Sedentary (3)': 3}
    cholestrol_mapping = {'High (0)': 0, 'BorderLine (1)': 1, 'Normal (2)': 2}

    # Query the model
    if st.button("Predict"):
        q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
            'age': age_mapping[age],
            'Gender': gender_mapping[gender],
            'Family': family_mapping[family_history],
            'diet': diet_mapping[diet],
            'Lifestyle': lifestyle_mapping[lifestyle],
            'cholestrol': cholestrol_mapping[cholestrol]
        })

        # Display the prediction
        st.write("Prediction for Heart Disease:")
        st.write(q['heartdisease'])
else:
    st.write("Please upload a CSV file to proceed.")
