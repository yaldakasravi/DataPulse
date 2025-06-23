# app.py

import streamlit as st
import pandas as pd
from utils.logger import log_usage

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="DataPulse - AutoML & EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# App Title and Description
# --------------------------
st.title("📊 DataPulse: Real-Time EDA & AutoML")
st.markdown("""
Welcome to **DataPulse**, a smart dashboard that lets you:
- Instantly explore datasets with automated visual analysis (AutoViz)
- Train and compare models for classification or regression (PyCaret)
- Understand your models using SHAP & LIME explainability tools
- Track runs using MLflow

👉 Start by uploading a CSV file.
""")

# --------------------------
# Logging
# --------------------------
log_usage("App launched")

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df
        st.success("✅ File uploaded and loaded successfully!")

        # Display preview
        with st.expander("📋 Preview uploaded dataset"):
            st.dataframe(df.head())

        st.info("➡️ Navigate to the sidebar and select **EDA**, **AutoML**, or **Explainability**.")

    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
else:
    st.warning("⚠️ Please upload a CSV file to get started.")

# --------------------------
# Footer
# --------------------------
st.markdown("""
---
🔧 Built with [Streamlit](https://streamlit.io/) · 📦 Powered by `pandas`, `AutoViz`, `PyCaret`, `SHAP`, `LIME`, and `MLflow`.
""")
