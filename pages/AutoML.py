import streamlit as st
import pandas as pd
from utils.automl_utils import run_pycaret_classification, run_pycaret_regression

# ----------------------------
# Page Title
# ----------------------------
st.title(" AutoML - Classification & Regression")

# ----------------------------
# Check for Uploaded Data
# ----------------------------
if "data" not in st.session_state:
    st.warning(" No data found. Please upload a CSV file on the main page.")
    st.stop()

# ----------------------------
# Load Data
# ----------------------------
df = st.session_state["data"]

# ----------------------------
# Task Selection UI
# ----------------------------
st.markdown("###  Step 1: Choose Your Task")
task = st.radio("Select the type of machine learning task:", ["Classification", "Regression"])

# ----------------------------
# Target Selection UI
# ----------------------------
st.markdown("###  Step 2: Choose Your Target Variable")
target = st.selectbox("Select the column you want to predict:", df.columns)

# ----------------------------
# Run AutoML
# ----------------------------
if st.button(" Run AutoML"):
    if not target:
        st.warning("Please select a target column.")
        st.stop()

    with st.spinner(f"Running PyCaret for {task}... "):
        try:
            if task == "Classification":
                best_model, model_df = run_pycaret_classification(df, target)
            else:
                best_model, model_df = run_pycaret_regression(df, target)

            # ----------------------------
            # Show Leaderboard and Model Info
            # ----------------------------
            st.success(" AutoML completed successfully!")
            st.markdown("###  Model Leaderboard")
            st.dataframe(model_df)

            st.markdown("###  Best Model Saved")
            st.info("The best model has been saved as `best_model.pkl` and is available for explainability.")

        except Exception as e:
            st.error(f" AutoML failed: {e}")
