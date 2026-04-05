from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from api.core.config import settings
from api.handlers.router import router
from api.middleware.security import RateLimitMiddleware, ApiKeyMiddleware
from api import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    from pipeline.train_pipeline import resolve_mlflow
    resolve_mlflow()
    logger.info(f"Starting up | ENV={settings.ENV} | MLflow={settings.MLFLOW_URI}")
    yield
    logger.info("Shutting down ...")


app = FastAPI(
    lifespan    = lifespan
    , title     = "Irrigation MLOps API"
    , version   = "1.0.0"
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(
    CORSMiddleware
    , allow_origins     = ["*"]
    , allow_methods     = ["*"]
    , allow_headers     = ["*"]
)

app.include_router(router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    return JSONResponse(
        status_code = exc.status_code
        , content   = {
            "status"    : exc.status_code
            , "message" : exc.detail
            , "data"    : None
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app"
        , host      = "0.0.0.0"
        , port      = settings.API_PORT
        , reload    = settings.ENV == "dev"
    )
