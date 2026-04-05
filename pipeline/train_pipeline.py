import os
import json
import yaml
import pickle
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from pipeline.preprocess_pipeline import PreprocessResult

from api import logger

def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def resolve_mlflow():
    from api.core.config import settings

    uri        = os.environ.get("MLFLOW_TRACKING_URI") or settings.MLFLOW_URI
    user       = os.environ.get("MLFLOW_TRACKING_USERNAME") or settings.MLFLOW_TRACKING_USERNAME
    password   = os.environ.get("MLFLOW_TRACKING_PASSWORD") or settings.MLFLOW_TRACKING_PASSWORD
    experiment = os.environ.get("MLFLOW_EXPERIMENT") or settings.MLFLOW_EXPERIMENT
    endpoint   = os.environ.get("MINIO_ENDPOINT") or settings.MINIO_ENDPOINT
    access_key = os.environ.get("MINIO_ACCESS_KEY") or settings.MINIO_ACCESS_KEY
    secret_key = os.environ.get("MINIO_SECRET_KEY") or settings.MINIO_SECRET_KEY

    os.environ["MLFLOW_TRACKING_USERNAME"]  = user
    os.environ["MLFLOW_TRACKING_PASSWORD"]  = password
    os.environ["MLFLOW_S3_ENDPOINT_URL"]    = endpoint
    os.environ["AWS_ACCESS_KEY_ID"]         = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"]     = secret_key

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def build_model(params: dict):
    # Loading training config pipeline

    train_cfg   = params["train"]
    choice      = train_cfg["model"]
    cfg         = train_cfg[choice]
    seed        = train_cfg["seed"]

    if choice == "lgbm":
        return LGBMClassifier(
            n_estimators    = cfg["n_estimators"]
            , learning_rate = cfg["learning_rate"]
            , class_weight  = cfg["class_weight"]
            , random_state  = seed
        )
    
    if choice == "xgb":
        return XGBClassifier(
            n_estimators    = cfg["n_estimators"]
            , learning_rate = cfg["learning_rate"]
            , random_state  = seed
            , device        = "cpu"
        )
    
    if choice == "catboost":
        return CatBoostClassifier(
            n_estimators        = cfg["n_estimators"]
            , learning_rate     = cfg["learning_rate"]
            , auto_class_weights = "Balanced"
            , train_dir         = cfg["train_dir"]
            , random_state      = seed
            , verbose           = 0
        )
    raise ValueError(f"Unknown model: {choice}")


def extract_feature_importance(model, feature_names: list[str], choice: str) -> dict[str, float]:
    try:
        if choice == "lgbm":
            imp = model.feature_importances_
        elif choice == "xgb":
            imp = model.feature_importances_
        elif choice == "catboost":
            imp = model.get_feature_importance()
        else:
            return {}

        total = float(imp.sum()) or 1.0
        return {name: round(v.item() / total, 6) for name, v in zip(feature_names, imp)}
    except Exception:
        return {}


def run(params: dict):
    with open("models/preprocessor.pkl", "rb") as f:
        pre: PreprocessResult = pickle.load(f)

    model   = build_model(params)
    choice  = params["train"]["model"]
    cfg     = params["train"][choice]

    fit_kwargs = {}
    if choice == "xgb":
        fit_kwargs["sample_weight"] = pre.sample_weights

    resolve_mlflow()

    Path("models").mkdir(exist_ok=True)

    with mlflow.start_run() as active_run:
        model.fit(pre.X_train, pre.y_train, **fit_kwargs)

        val_score   = balanced_accuracy_score(pre.y_val, model.predict(pre.X_val))
        run_id      = active_run.info.run_id

        feature_names   = list(pre.X_train.columns)
        importance      = extract_feature_importance(model, feature_names, choice)

        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open("models/feature_importance.json", "w") as f:
            json.dump(importance, f)

        mlflow.log_param("model", choice)
        mlflow.log_metric("balanced_accuracy_val", val_score)
        mlflow.log_dict(importance, "feature_importance.json")

        mlflow.sklearn.log_model(
            sk_model        = model
            , name          = choice
            , params        = cfg
            , input_example = pre.X_train.iloc[:5]
        )

    logger.info(f"Model : {choice}")
    logger.info(f"Val balanced accuracy : {val_score:.4f}")
    logger.info(f"MLflow run_id         : {run_id}")

    Path("models/run_id").write_text(run_id)


if __name__ == "__main__":
    params = load_params()
    Path(params["artifacts"]["model_dir"]).mkdir(exist_ok=True)
    run(params)
