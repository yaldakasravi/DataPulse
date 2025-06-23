# 📊 DataPulse

**Real-time Exploratory Data Analysis (EDA) and AutoML Dashboard**

---

## 🚀 Project Overview

DataPulse is an interactive web application that enables users to:

- Instantly upload CSV datasets and perform **automated EDA** using **AutoViz**.
- Build and compare machine learning models for **classification** and **regression** with **PyCaret AutoML**.
- Visualize model explanations through **SHAP** and **LIME** for transparency and interpretability.
- Track experiments and model versions via **MLflow**.
- Export reports and models for further analysis or deployment.

The app is built with **Streamlit** for a seamless user experience and deployed on platforms like **Streamlit Cloud** or **AWS EC2**.

---

## 📦 Features

- Easy CSV upload and preview
- Automated visual EDA with charts and insights
- AutoML with leaderboard comparison and performance tuning
- Model explainability with SHAP and LIME visualizations
- Experiment tracking with MLflow integration
- Exportable model and reports

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning repo)

### Setup Steps

1. Clone this repository:

```bash
git clone https://github.com/yaldakasravi/DataPulse.git
cd DataPulse
Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install required Python packages:
pip install -r requirements.txt
Copy .env.example to .env and fill in any required secrets (e.g., MLflow URI, AWS keys):
cp .env.example .env
🚀 Running the App

Start the Streamlit app locally:

streamlit run app.py
Open the displayed URL (usually http://localhost:8501) in your browser.

🗂️ Project Structure

├── app.py                 # Main entry point
├── pages/                 # Streamlit multi-page app
│   ├── 1_🏠_Home.py
│   ├── 2_📊_EDA.py
│   ├── 3_🤖_AutoML.py
│   └── 4_📈_Explainability.py
├── utils/                 # Helper utility modules
│   ├── automl_utils.py
│   ├── eda_utils.py
│   ├── mlflow_utils.py
│   ├── logger.py
│   └── shaplime_utils.py
├── .streamlit/            # Streamlit configuration
│   └── config.toml
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
🧰 Dependencies

Streamlit
pandas
AutoViz
PyCaret
SHAP
LIME
MLflow
