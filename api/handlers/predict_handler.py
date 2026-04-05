from fastapi import APIRouter, HTTPException

from api.schemas.predict_schema import PredictRequest, PredictResponse
from api.schemas.base_schema import RESPONSE_SCHEMA
from pipeline.predict_pipeline import load_artifacts, predict
from api import logger


router = APIRouter(prefix="/predict", tags=["Predict"])

_model  = None
_pre    = None


def get_artifacts():
    global _model, _pre
    if _model is None or _pre is None:
        _model, _pre = load_artifacts()
    return _model, _pre


@router.post("", response_model=RESPONSE_SCHEMA[PredictResponse])
async def predict_endpoint(req: PredictRequest):
    try:
        model, pre  = get_artifacts()
        result      = predict(req.model_dump(), model, pre)

        logger.info(f"Predicted: {result['irrigation_need']} ({result['confidence']:.2%})")

        return RESPONSE_SCHEMA(
            status      = 200
            , message   = "Prediction successful"
            , data      = PredictResponse(**result)
        )

    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run /train first.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
