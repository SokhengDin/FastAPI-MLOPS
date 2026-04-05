import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


JobStatus = Literal["pending", "running", "done", "failed"]


@dataclass
class Job:
    id          : str
    status      : JobStatus         = "pending"
    model       : str               = ""
    logs        : list[str]         = field(default_factory=list)
    run_id      : str | None        = None
    score       : float | None      = None
    error       : str | None        = None
    started_at  : str | None        = None
    finished_at : str | None        = None


_store: dict[str, Job] = {}


def create_job(model: str) -> Job:
    job             = Job(id=str(uuid.uuid4()), model=model, started_at=datetime.now(timezone.utc).isoformat())
    _store[job.id]  = job
    return job


def get_job(job_id: str) -> Job | None:
    return _store.get(job_id)


def list_jobs() -> list[Job]:
    return sorted(_store.values(), key=lambda j: j.started_at or "", reverse=True)
