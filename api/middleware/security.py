import time

from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from api.core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):

    def __init__(self, app):
        super().__init__(app)
        self._hits   : dict = defaultdict(list)
        self._window : int  = 60

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.endswith("/predict"):
            return await call_next(request)

        ip  = request.client.host
        now = time.time()

        self._hits[ip] = [t for t in self._hits[ip] if now - t < self._window]

        if len(self._hits[ip]) >= settings.RATE_LIMIT_PER_MIN:
            raise HTTPException(
                status_code = 429
                , detail    = f"Rate limit exceeded. Max {settings.RATE_LIMIT_PER_MIN} requests/min."
            )

        self._hits[ip].append(now)
        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.endswith("/train"):
            return await call_next(request)

        key = request.headers.get("x-api-key", "")

        if not settings.API_KEY or key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

        return await call_next(request)
