

import streamlit as st
import pandas as pd
import os
from utils.eda_utils import run_autoviz

# ----------------------------
# Page Title
# ----------------------------
st.title(" Exploratory Data Analysis (EDA)")

# ----------------------------
# Check if data is uploaded
# ----------------------------
if "data" not in st.session_state:
    st.warning(" No data found. Please upload a CSV file on the main page.")
    st.stop()

# ----------------------------
# Load Data from Session
# ----------------------------
df = st.session_state["data"]

# ----------------------------
# Show Data Preview
# ----------------------------
with st.expander(" Preview of Uploaded Data"):
    st.write(df.head())

# ----------------------------
# Run AutoViz for EDA
# ----------------------------
st.subheader(" Automated Visualizations")

temp_csv_path = "temp_uploaded.csv"
df.to_csv(temp_csv_path, index=False)

with st.spinner("Generating visualizations with AutoViz... "):
    try:
        run_autoviz(temp_csv_path)
    except Exception as e:
        st.error(f" AutoViz failed: {e}")
    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

st.info(" Tip: Use this analysis to inform feature selection and model choices.")
