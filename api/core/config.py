from pydantic_settings import BaseSettings
from decouple import config


class Settings(BaseSettings):

    API_V1_PREFIX   : str   = "/api/v1"
    API_BASE_URL    : str   = config("API_BASE_URL",    cast=str,   default="http://localhost")
    API_PORT        : int   = config("API_PORT",        cast=int,   default=8000)

    ENV             : str   = config("ENV",             cast=str,   default="development")

    MLFLOW_URI                  : str   = config("MLFLOW_URI",                  cast=str,   default="http://localhost:5001")
    MLFLOW_EXPERIMENT           : str   = config("MLFLOW_EXPERIMENT",           cast=str,   default="irrigation-mlops")
    MLFLOW_TRACKING_USERNAME    : str   = config("MLFLOW_TRACKING_USERNAME",    cast=str,   default="")
    MLFLOW_TRACKING_PASSWORD    : str   = config("MLFLOW_TRACKING_PASSWORD",    cast=str,   default="")

    MINIO_ENDPOINT              : str   = config("MINIO_ENDPOINT",              cast=str,   default="")
    MINIO_ACCESS_KEY            : str   = config("MINIO_ACCESS_KEY",            cast=str,   default="")
    MINIO_SECRET_KEY            : str   = config("MINIO_SECRET_KEY",            cast=str,   default="")

    API_KEY             : str   = config("API_KEY",             cast=str,   default="")
    RATE_LIMIT_PER_MIN  : int   = config("RATE_LIMIT_PER_MIN",  cast=int,   default=20)


settings = Settings()
