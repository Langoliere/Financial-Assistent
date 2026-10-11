"""Middleware безопасности, лимитов и трассировки HTTP-запросов."""

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from investbot.observability import HTTP_LATENCY, HTTP_REQUESTS, request_id_context

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """Потокобезопасный лимитер запросов в памяти одного процесса."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: dict[str, deque[float]] = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        """Проверить и зарегистрировать запрос для указанного ключа."""
        now = time.monotonic()
        async with self.lock:
            timestamps = self.requests[key]
            while timestamps and now - timestamps[0] >= self.window_seconds:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(now)
            return True


class ProductionMiddleware(BaseHTTPMiddleware):
    """Применять request ID, API key, rate limit и защитные HTTP-заголовки."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.api_key = os.getenv("INVESTBOT_API_KEY")
        self.max_body_bytes = int(os.getenv("MAX_REQUEST_BODY_BYTES", "262144"))
        self.rate_limiter = SlidingWindowRateLimiter(
            int(os.getenv("PREDICTION_RATE_LIMIT_PER_MINUTE", "30"))
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Проверить запрос и записать технические метрики ответа."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
        token = request_id_context.set(request_id)
        started = time.perf_counter()
        route = request.url.path
        response: Response
        try:
            rejection = await self._reject_request(request)
            response = rejection if rejection else await call_next(request)
        except Exception:
            logger.exception("Необработанная ошибка HTTP-запроса")
            response = JSONResponse(
                status_code=500,
                content={"detail": "Внутренняя ошибка сервиса", "request_id": request_id},
            )
        finally:
            request_id_context.reset(token)

        duration = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method, route).observe(duration)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'"
        )
        return response

    async def _reject_request(self, request: Request) -> Response | None:
        """Вернуть ранний отказ при нарушении ограничений безопасности."""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_bytes:
            return JSONResponse(status_code=413, content={"detail": "Тело запроса слишком велико"})

        is_prediction = request.url.path in {
            "/v1/predict/direction",
            "/api/ui/predict",
        }
        if is_prediction:
            client = request.client.host if request.client else "unknown"
            if not await self.rate_limiter.allow(client):
                return JSONResponse(status_code=429, content={"detail": "Лимит запросов превышен"})

        if self.api_key and request.url.path.startswith("/v1/"):
            supplied = request.headers.get("X-API-Key")
            if supplied != self.api_key:
                return JSONResponse(status_code=401, content={"detail": "Неверный API key"})
        return None
