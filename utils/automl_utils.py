
from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models, save_model as clf_save_model, pull as clf_pull, load_model as clf_load_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, save_model as reg_save_model, pull as reg_pull, load_model as reg_load_model

import pandas as pd

def run_pycaret_classification(df: pd.DataFrame, target: str):
    """
    Runs PyCaret classification AutoML pipeline.

    Args:
        df (pd.DataFrame): Dataset including target variable.
        target (str): Target column name.

    Returns:
        tuple: (best_model, leaderboard_df)
    """
    clf_setup(data=df, target=target, silent=True, verbose=False, session_id=42, log_experiment=False)
    best_model = clf_compare_models()
    leaderboard_df = clf_pull()
    clf_save_model(best_model, 'best_model')
    return best_model, leaderboard_df

def run_pycaret_regression(df: pd.DataFrame, target: str):
    """
    Runs PyCaret regression AutoML pipeline.

    Args:
        df (pd.DataFrame): Dataset including target variable.
        target (str): Target column name.

    Returns:
        tuple: (best_model, leaderboard_df)
    """
    reg_setup(data=df, target=target, silent=True, verbose=False, session_id=42, log_experiment=False)
    best_model = reg_compare_models()
    leaderboard_df = reg_pull()
    reg_save_model(best_model, 'best_model')
    return best_model, leaderboard_df
