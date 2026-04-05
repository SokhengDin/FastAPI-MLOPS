import json
import yaml

from pathlib import Path
from fastapi import APIRouter, HTTPException

from api.schemas.base_schema import RESPONSE_SCHEMA
from api import logger


router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("", response_model=RESPONSE_SCHEMA[dict])
async def get_metrics():
    try:
        path = Path("models/metrics.json")

        if not path.exists():
            raise HTTPException(status_code=404, detail="No metrics found. Run training first.")

        with open(path) as f:
            metrics = json.load(f)

        return RESPONSE_SCHEMA(
            status      = 200
            , message   = "OK"
            , data      = metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance", response_model=RESPONSE_SCHEMA[dict])
async def get_feature_importance():
    try:
        path = Path("models/feature_importance.json")

        if not path.exists():
            raise HTTPException(status_code=404, detail="No feature importance found. Run training first.")

        with open(path) as f:
            import json
            data = json.load(f)

        return RESPONSE_SCHEMA(status=200, message="OK", data=data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info", response_model=RESPONSE_SCHEMA[dict])
async def get_model_info():
    try:
        run_id_path = Path("models/run_id")
        params_path = Path("params.yaml")

        if not params_path.exists():
            raise HTTPException(status_code=404, detail="params.yaml not found.")

        with open(params_path) as f:
            params = yaml.safe_load(f)

        info = {
            "model"         : params["train"]["model"]
            , "run_id"      : run_id_path.read_text().strip() if run_id_path.exists() else None
            , "mlflow_uri"  : params["mlflow"]["tracking_uri"]
            , "experiment"  : params["mlflow"]["experiment_name"]
        }

        return RESPONSE_SCHEMA(
            status      = 200
            , message   = "OK"
            , data      = info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
