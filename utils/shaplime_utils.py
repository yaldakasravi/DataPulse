# utils/shaplime_utils.py

import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
from pycaret.classification import load_model as load_classification_model
from pycaret.regression import load_model as load_regression_model
import numpy as np

def explain_shap(model, df: pd.DataFrame):
    """
    Generates a SHAP summary plot figure for the given model and dataframe.

    Args:
        model: Trained PyCaret model object.
        df (pd.DataFrame): Dataset to explain (features only).

    Returns:
        matplotlib.figure.Figure: The SHAP summary plot figure.
    """
    # Prepare data - exclude target if present
    if hasattr(model, 'target'):
        X = df.drop(columns=[model.target], errors='ignore')
    else:
        X = df.copy()

    # Create SHAP explainer
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    # Plot SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig

def explain_lime(model, df: pd.DataFrame):
    """
    Generates a LIME explanation figure for the first instance in the dataframe.

    Args:
        model: Trained PyCaret model object.
        df (pd.DataFrame): Dataset to explain (features only).

    Returns:
        matplotlib.figure.Figure: The LIME explanation plot figure.
    """
    # Prepare data - exclude target if present
    if hasattr(model, 'target'):
        X = df.drop(columns=[model.target], errors='ignore')
    else:
        X = df.copy()

    # Initialize LimeTabularExplainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns,
        mode='classification' if "Classifier" in str(type(model)) else 'regression',
        discretize_continuous=True
    )

    # Explain the first instance
    instance = X.iloc[0].values
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba if hasattr(model, "predict_proba") else model.predict,
        num_features=min(10, X.shape[1])
    )

    # Plot LIME explanation
    fig = explanation.as_pyplot_figure()
    plt.tight_layout()
    return fig
