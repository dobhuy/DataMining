import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os
from lightgbm import LGBMClassifier


st.set_page_config(layout="wide")

# Title
st.markdown(
    """
    <h1 style="text-align: center;">Predicting Depression to Enhance Mental Health Interventions</h1>
    """,
    unsafe_allow_html=True
)

# --- Layout using Tabs ---
tab1 = st.tabs(["📋 Form Input"])

# --- Tab 1: Form Input ---
# --- Tab 1: Form Input ---
with tab1:
    # col1, col2, col3 = st.columns([2, 1, 1], gap="large")
    col1, col2 = st.columns(2)

    # Load and display training dataset
    with col1:
        try:
            path_train = '../Data/data_processed.csv'  # Update this to the correct path if necessary
            df = pd.read_csv(path_train)

            path_test = '../Data/test.csv'
            df_test = pd.read_csv(path_test)
            
            # Load test data
            st.markdown(
            """
            <h3 style="text-align: center;">Samples Test</h3>
            """,
            unsafe_allow_html=True
            )
            st.dataframe(df_test)  # Display the first 10 rows of the dataset
        except Exception as e:
            st.error("Failed to load dataset. Please check the file path.")

    # Enter Input
    with col2:
        col2_1, col2_2 = st.columns(2)
        # st.markdown(
        #     """
        #     <h3 style="text-align: center;">📋 Input Form</h3>
        #     """,
        #     unsafe_allow_html=True
        # )
        with col2_1:
            gender = st.multiselect("**Gender**", df['Gender'].unique())
            city = st.multiselect("**City**", df['City'].unique())
            working = st.multiselect("**Working Professional or Student**", df['Working Professional or Student'].unique())
            profession = st.multiselect("**Profession**", df['Profession'].unique())
            sleep = st.multiselect("**Sleep Duration**", df['Sleep Duration'].unique())
            age = st.slider("**Age**", min_value=18, max_value=60, value=25)
            work_pressure = st.slider("**Work Pressure**", min_value=1, max_value=5, value=3)

        with col2_2:
            habit = st.multiselect("**Dietary Habits**", df['Dietary Habits'].unique())
            degree = st.multiselect("**Degree**", df['Degree'].unique())
            thoughts = st.multiselect("**Have you ever had suicidal thoughts ?**", df['Have you ever had suicidal thoughts ?'].unique())
            history = st.multiselect("**Family History of Mental Illness**", df['Family History of Mental Illness'].unique())
            satisfaction = st.slider("**Job Satisfaction**", min_value=1, max_value=5, value=4)
            work_hours = st.slider("**Work/Study Hours (per day)**", min_value=0, max_value=12, value=8)
            stress = st.slider("**Financial Stress**", min_value=1, max_value=5, value=3)

        # Submit button
        predict_button = st.button("🚀 Predict")

        if predict_button:
            # Load encoders, models
            lgbm_clf = joblib.load('../Weights/best_model.joblib')
            encoders = joblib.load('../Encoders/label_encoders.joblib')

            # Create Input
            columns = ['Gender', 'Age', 'City', 'Working Professional or Student', 'Profession', 'Work Pressure', 
                    'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Degree', 
                    'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 
                    'Family History of Mental Illness']
            data_input = [gender[0], age, city[0], working[0], profession[0], work_pressure, satisfaction,
                        sleep[0], habit[0], degree[0], thoughts[0], work_hours, stress, history[0]]

            x_test = pd.DataFrame([data_input], columns=columns)

            # Using encoders
            for name, encoder in encoders.items():
                x_test[name] = encoder.transform(x_test[name])
            
            # Predict
            yhat = lgbm_clf.predict(x_test)
            
            # Mapping
            id_to_name = {
                0: "No Depression",
                1: "Depression"
            }

            # Display predicted and true results
            predicted_result = id_to_name[yhat[0]]

            st.markdown(
            f"<h3 style='color: red;'>{predicted_result}</h3>",
            unsafe_allow_html=True
            )

           
