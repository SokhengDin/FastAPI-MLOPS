from pydantic import BaseModel
from typing import Optional


class TrainRequest(BaseModel):
    model       : Optional[str]     = "lgbm"
    test_size   : Optional[float]   = 0.2
    seed        : Optional[int]     = 42


class TrainJobResponse(BaseModel):
    job_id      : str
    status      : str
    message     : str


class TrainStatusResponse(BaseModel):
    job_id      : str
    status      : str
    model       : str
    logs        : list[str]
    run_id      : Optional[str]     = None
    score       : Optional[float]   = None
    error       : Optional[str]     = None


class TrainHistoryItem(BaseModel):
    job_id      : str
    model       : str
    status      : str
    score       : Optional[float]   = None
    run_id      : Optional[str]     = None
    error       : Optional[str]     = None
    started_at  : Optional[str]     = None
    finished_at : Optional[str]     = None
