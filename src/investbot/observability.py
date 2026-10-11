"""Структурированное логирование и Prometheus-метрики сервиса."""

import json
import logging
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

request_id_context: ContextVar[str] = ContextVar("request_id", default="-")

HTTP_REQUESTS = Counter(
    "investbot_http_requests_total",
    "Количество HTTP-запросов",
    ("method", "path", "status"),
)
HTTP_LATENCY = Histogram(
    "investbot_http_request_duration_seconds",
    "Длительность HTTP-запросов",
    ("method", "path"),
)
PREDICTION_LATENCY = Histogram(
    "investbot_prediction_duration_seconds",
    "Длительность инференса",
)
MOEX_LATENCY = Histogram(
    "investbot_moex_request_duration_seconds",
    "Длительность получения данных MOEX",
)
PREDICTIONS = Counter(
    "investbot_predictions_total",
    "Количество предсказаний",
    ("direction", "model_version"),
)
DATA_QUALITY_FAILURES = Counter(
    "investbot_data_quality_failures_total",
    "Количество ошибок качества данных",
    ("reason",),
)
DRIFT_SCORE = Gauge(
    "investbot_data_drift_score",
    "Максимальное нормированное отклонение входных данных",
)
FEATURE_DRIFT_SCORE = Gauge(
    "investbot_feature_drift_score",
    "Максимальное нормированное отклонение признаков",
)
PREDICTION_PROBABILITY = Gauge(
    "investbot_prediction_probability_up",
    "Последняя вероятность роста",
)
MODEL_READY = Gauge(
    "investbot_model_ready",
    "Готовность модели к инференсу",
)


class JsonFormatter(logging.Formatter):
    """Форматировать записи стандартного logging в JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Сформировать JSON-запись без сериализации внутренних полей logging."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_context.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Настроить корневой логгер на структурированный JSON-вывод."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


class Timer:
    """Простой монотонный таймер для измерения внешних операций."""

    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def seconds(self) -> float:
        """Вернуть прошедшее время в секундах."""
        return time.perf_counter() - self.started
