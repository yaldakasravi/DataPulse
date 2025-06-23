# pages/4_📈_Explainability.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.classification import load_model as load_classification_model
from pycaret.regression import load_model as load_regression_model
from utils.shaplime_utils import explain_shap, explain_lime

st.title("📈 Model Explainability with SHAP & LIME")

# ----------------------------
# Check for Uploaded Data
# ----------------------------
if "data" not in st.session_state:
    st.warning("⚠️ Please upload a CSV file and run AutoML first.")
    st.stop()

df = st.session_state["data"]

# ----------------------------
# Select Model Type
# ----------------------------
model_type = st.radio("Select model type to explain:", ("Classification", "Regression"))

# ----------------------------
# Load the Best Model
# ----------------------------
try:
    if model_type == "Classification":
        model = load_classification_model("best_model")
    else:
        model = load_regression_model("best_model")
    st.success(f" Loaded saved {model_type} model successfully.")
except Exception as e:
    st.error(f" Failed to load model: {e}")
    st.stop()

# ----------------------------
# Select Explainability Method
# ----------------------------
explain_method = st.selectbox("Choose explainability method:", ["SHAP", "LIME"])

# ----------------------------
# Generate Explanation
# ----------------------------
if st.button("Generate Explanation"):
    with st.spinner(f"Running {explain_method} explanation... ⏳"):
        try:
            if explain_method == "SHAP":
                fig = explain_shap(model, df)
            else:
                fig = explain_lime(model, df)

            st.pyplot(fig)
        except Exception as e:
            st.error(f" Explanation failed: {e}")
else:
    st.info("Press the button above to generate model explanation.")
