
import mlflow
import mlflow.sklearn
import pickle
import json
import yaml

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException
from sklearn.metrics import balanced_accuracy_score

from api.schemas.train_schema import TrainRequest, TrainJobResponse, TrainStatusResponse, TrainHistoryItem
from api.schemas.base_schema import RESPONSE_SCHEMA
from api.core.job_store import create_job, get_job, list_jobs, Job
from sklearn.metrics import classification_report
from pipeline.preprocess_pipeline import run as preprocess_run
from pipeline.train_pipeline import build_model, resolve_mlflow, extract_feature_importance
from api import logger


router      = APIRouter(prefix="/train", tags=["Train"])
_executor   = ThreadPoolExecutor(max_workers=2)


def _run_training(job: Job, params: dict):
    try:
        job.status = "running"
        job.logs.append("Preprocessing data...")

        pre = preprocess_run(params)
        job.logs.append(f"Data ready — train: {pre.X_train.shape[0]} rows, val: {pre.X_val.shape[0]} rows")

        model   = build_model(params)
        choice  = job.model
        cfg     = params["train"][choice]
        job.logs.append(f"Built {choice.upper()} model, starting training...")

        fit_kwargs = {}
        if choice == "xgb":
            fit_kwargs["sample_weight"] = pre.sample_weights

        resolve_mlflow()

        Path("models").mkdir(exist_ok=True)

        with mlflow.start_run() as active_run:
            model.fit(pre.X_train, pre.y_train, **fit_kwargs)
            job.logs.append("Training complete, evaluating...")

            preds           = model.predict(pre.X_val)
            score           = balanced_accuracy_score(pre.y_val, preds)
            report          = classification_report(pre.y_val, preds, target_names=pre.le_target.classes_, output_dict=True)
            run_id          = active_run.info.run_id

            feature_names   = list(pre.X_train.columns)
            importance      = extract_feature_importance(model, feature_names, choice)

            with open("models/model.pkl", "wb") as f:
                pickle.dump(model, f)

            with open("models/preprocessor.pkl", "wb") as f:
                pickle.dump(pre, f)

            with open("models/feature_importance.json", "w") as f:
                json.dump(importance, f)

            with open("models/metrics.json", "w") as f:
                json.dump({"balanced_accuracy": score, "classification_report": report}, f, default=float)

            with open("models/run_id", "w") as f:
                f.write(run_id)

            mlflow.log_param("model", choice)
            mlflow.log_metric("balanced_accuracy_val", float(score))
            mlflow.log_dict(importance, "feature_importance.json")
            for cls, vals in report.items():
                if isinstance(vals, dict):
                    for metric, val in vals.items():
                        mlflow.log_metric(f"{cls}_{metric.replace('-', '_')}", float(val))

            job.logs.append("Logging model to MLflow...")
            mlflow.sklearn.log_model(
                sk_model        = model
                , name          = choice
                , params        = cfg
                , input_example = pre.X_train.iloc[:5]
            )

        job.score       = round(score, 4)
        job.run_id      = run_id
        job.status      = "done"
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.logs.append(f"Done — balanced accuracy: {score:.4f} | run_id: {run_id[:8]}")

        logger.info(f"Job {job.id[:8]} done | {job.model} | score={score:.4f}")

    except Exception as e:
        job.status      = "failed"
        job.error       = str(e)
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.logs.append(f"Failed: {e}")
        logger.error(f"Job {job.id[:8]} failed: {e}")


@router.post("", response_model=RESPONSE_SCHEMA[TrainJobResponse])
async def train(req: TrainRequest):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    params["preprocess"]["test_size"]   = req.test_size
    params["preprocess"]["seed"]        = req.seed
    params["train"]["model"]            = req.model

    job = create_job(req.model)
    _executor.submit(_run_training, job, params)

    logger.info(f"Training job {job.id[:8]} queued | model={req.model}")

    return RESPONSE_SCHEMA(
        status      = 202
        , message   = "Training started"
        , data      = TrainJobResponse(
            job_id      = job.id
            , status    = job.status
            , message   = f"Training {req.model.upper()} in background"
        )
    )


@router.get("/history", response_model=RESPONSE_SCHEMA[list[TrainHistoryItem]])
async def train_history():
    jobs = list_jobs()
    return RESPONSE_SCHEMA(
        status      = 200
        , message   = "OK"
        , data      = [
            TrainHistoryItem(
                job_id      = j.id
                , model     = j.model
                , status    = j.status
                , score     = j.score
                , run_id    = j.run_id
                , error     = j.error
                , started_at  = j.started_at
                , finished_at = j.finished_at
            )
            for j in jobs
        ]
    )


@router.get("/{job_id}/status", response_model=RESPONSE_SCHEMA[TrainStatusResponse])
async def train_status(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return RESPONSE_SCHEMA(
        status      = 200
        , message   = job.status
        , data      = TrainStatusResponse(
            job_id      = job.id
            , status    = job.status
            , model     = job.model
            , logs      = job.logs
            , run_id    = job.run_id
            , score     = job.score
            , error     = job.error
        )
    )
