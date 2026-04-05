import os
import yaml
import json
import pickle
import mlflow

from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score
    , classification_report
)

from pipeline.preprocess_pipeline import PreprocessResult

from api import logger


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def resolve_mlflow():
    from api.core.config import settings

    uri      = os.environ.get("MLFLOW_TRACKING_URI") or settings.MLFLOW_URI
    user     = os.environ.get("MLFLOW_TRACKING_USERNAME") or settings.MLFLOW_TRACKING_USERNAME
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD") or settings.MLFLOW_TRACKING_PASSWORD
    experiment = os.environ.get("MLFLOW_EXPERIMENT") or settings.MLFLOW_EXPERIMENT

    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def run(params: dict):
    with open("models/preprocessor.pkl", "rb") as f:
        pre: PreprocessResult = pickle.load(f)

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    preds   = model.predict(pre.X_val)
    score   = balanced_accuracy_score(pre.y_val, preds)
    report  = classification_report(
        pre.y_val
        , preds
        , target_names  = pre.le_target.classes_
        , output_dict   = True
    )

    metrics = {
        "balanced_accuracy"       : score
        , "classification_report" : report
    }

    with open(params["artifacts"]["metrics_file"], "w") as f:
        json.dump(metrics, f, indent=2)

    resolve_mlflow()

    # Resume the run started by train_pipeline so all metrics live in one run
    run_id_file = Path("models/run_id")
    run_id      = run_id_file.read_text().strip() if run_id_file.exists() else None

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("balanced_accuracy", score)
        for cls, vals in report.items():
            if isinstance(vals, dict):
                for metric, val in vals.items():
                    mlflow.log_metric(f"{cls}_{metric}", val)

    logger.info(f"Balanced accuracy: {score:.4f}")
    logger.info(classification_report(
        pre.y_val, preds, target_names=pre.le_target.classes_
    ))


if __name__ == "__main__":
    params = load_params()
    run(params)
