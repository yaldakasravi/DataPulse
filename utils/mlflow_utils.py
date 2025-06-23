
import mlflow
import mlflow.pycaret
import os

def start_mlflow_run(run_name: str = None):
    """
    Starts an MLflow run.

    Args:
        run_name (str): Optional name for the run.

    Returns:
        mlflow.ActiveRun: The active MLflow run object.
    """
    return mlflow.start_run(run_name=run_name)

def log_pycaret_model(model, model_name="best_model", experiment_name="DataPulse_Experiments", params=None, metrics=None):
    """
    Logs a PyCaret model, parameters, and metrics to MLflow.

    Args:
        model: Trained PyCaret model object.
        model_name (str): Name to register the model under.
        experiment_name (str): MLflow experiment name.
        params (dict): Dictionary of parameters to log.
        metrics (dict): Dictionary of metrics to log.

    Returns:
        None
    """
    # Set or create experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)

        # Log metrics
        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

        # Log PyCaret model
        mlflow.pycaret.log_model(model, artifact_path=model_name)

def end_mlflow_run():
    """
    Ends the current MLflow run.
    """
    mlflow.end_run()
