import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():
    st.title("Predicting Student Performance")

    # Create a form for user input
    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ["male", "female"])
        race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
        parental_level_of_education = st.selectbox("Parental Level of Education", [
            "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
        ])
        lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
        test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
        reading_score = st.number_input("Reading Score", min_value=0, max_value=100)
        writing_score = st.number_input("Writing Score", min_value=0, max_value=100)

        submit_button = st.form_submit_button(label="Predict")

    # Perform prediction when the form is submitted
    if submit_button:
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        pred_df = data.get_data_as_data_frame()
        st.write(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.write(f"Prediction: {results[0]}")

if __name__ == "__main__":
    main()
