import streamlit as st

# ----------------------------
# Page Title
# ----------------------------
st.title("🏠 Home - Welcome to DataPulse")

# ----------------------------
# App Introduction
# ----------------------------
st.markdown("""
Welcome to **DataPulse** — your all-in-one **Automated Machine Learning** and **Exploratory Data Analysis** dashboard.

This application allows you to:
- 🔍 Automatically visualize your dataset using **AutoViz**
- ⚙️ Train machine learning models using **PyCaret**
- 📈 Understand your models with **SHAP** and **LIME**
- 🧪 Track model versions and metrics with **MLflow**

---
""")

# ----------------------------
# Step-by-Step Instructions
# ----------------------------
st.header("📌 How to Use This App")

st.markdown("""
1. **Upload your dataset (CSV)** on the main screen.
2. Navigate to:
   - 🧪 **EDA** tab to view visualizations and statistics
   - 🤖 **AutoML** tab to train classification or regression models
   - 📈 **Explainability** tab to interpret your model
3. Download reports or export trained models (optional).
4. Check experiment logs using MLflow (if configured).

---

**Note:** The app runs entirely in your browser and does not store any of your uploaded data.
""")

# ----------------------------
# Tips or Notes
# ----------------------------
st.info("🚀 Tip: Start by uploading your CSV file on the **main page** (app homepage).")

# ----------------------------
# Optional Sidebar Navigation Message
# ----------------------------
st.sidebar.title("📂 Navigation")
st.sidebar.info("Use the sidebar to switch between EDA, AutoML, and Explainability sections.")

# ----------------------------
# Credits / Footer
# ----------------------------
st.markdown("""
---
Made with ❤️ by Yalda  
Powered by `Streamlit`, `AutoViz`, `PyCaret`, `SHAP`, and `LIME`
""")
