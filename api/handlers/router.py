from fastapi import APIRouter

from api.handlers.train_handler import router as train_router
from api.handlers.predict_handler import router as predict_router
from api.handlers.metrics_handler import router as metrics_router


router = APIRouter()

router.include_router(predict_router)
router.include_router(train_router)
router.include_router(metrics_router)
