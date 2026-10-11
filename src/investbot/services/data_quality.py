"""Проверки качества входных данных и мониторинг статистического дрейфа."""

import logging
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from investbot.observability import (
    DATA_QUALITY_FAILURES,
    DRIFT_SCORE,
    FEATURE_DRIFT_SCORE,
)

logger = logging.getLogger(__name__)


class DataQualityError(ValueError):
    """Ошибка, при которой входные данные нельзя безопасно использовать."""


class DataQualityMonitor:
    """Проверять OHLCV и сравнивать его статистики с обучающим профилем."""

    COLUMNS = ("open", "high", "low", "close", "volume", "value")

    def __init__(self, profile: dict[str, Any] | None = None) -> None:
        """Инициализировать монитор обучающим профилем и порогами окружения."""
        self.profile = profile or {}
        self.max_drift_score = float(os.getenv("MAX_DATA_DRIFT_SCORE", "5.0"))
        self.block_on_drift = os.getenv("BLOCK_ON_DATA_DRIFT", "false").lower() == "true"
        self.max_age_days = int(os.getenv("MAX_MARKET_DATA_AGE_DAYS", "7"))

    def validate(self, data: list[dict], check_freshness: bool = False) -> float:
        """Проверить схему, целостность, свежесть и вернуть drift score."""
        if len(data) < 40:
            self._fail("insufficient_history", "Требуется не менее 40 наблюдений")
        frame = pd.DataFrame(data)
        missing = set(self.COLUMNS).difference(frame.columns)
        if missing:
            self._fail("missing_columns", f"Отсутствуют колонки: {sorted(missing)}")

        try:
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
            frame[list(self.COLUMNS)] = frame[list(self.COLUMNS)].apply(
                pd.to_numeric, errors="raise"
            )
        except (TypeError, ValueError) as error:
            self._fail("invalid_types", "Некорректные типы входных данных", error)

        if frame["date"].duplicated().any():
            self._fail("duplicate_dates", "Обнаружены повторяющиеся даты")
        if not frame["date"].is_monotonic_increasing:
            self._fail("non_chronological", "Данные не отсортированы по времени")
        if not np.isfinite(frame[list(self.COLUMNS)].to_numpy()).all():
            self._fail("non_finite", "Обнаружены NaN или бесконечные значения")
        if (frame[["open", "high", "low", "close"]] <= 0).any().any():
            self._fail("non_positive_price", "Цены должны быть положительными")
        if (frame[["volume", "value"]] < 0).any().any():
            self._fail("negative_volume", "Объём и стоимость не могут быть отрицательными")
        if (
            (frame["high"] < frame[["open", "close"]].max(axis=1)).any()
            or (frame["low"] > frame[["open", "close"]].min(axis=1)).any()
            or (frame["high"] < frame["low"]).any()
        ):
            self._fail("invalid_ohlc", "Нарушены ограничения OHLC")

        if check_freshness:
            age = datetime.now(UTC) - frame["date"].iloc[-1].to_pydatetime()
            if age.days > self.max_age_days:
                self._fail("stale_data", f"Последняя свеча устарела на {age.days} дней")

        score = self._calculate_drift(frame)
        DRIFT_SCORE.set(score)
        if score > self.max_drift_score:
            logger.warning(
                "Обнаружен дрейф входных данных: score=%.3f threshold=%.3f",
                score,
                self.max_drift_score,
            )
            if self.block_on_drift:
                self._fail("data_drift", "Порог дрейфа входных данных превышен")
        return score

    def validate_feature_drift(self, features: pd.DataFrame) -> float:
        """Сравнить engineered features с профилем обучающей выборки."""
        score = self._normalized_mean_shift(
            features,
            self.profile.get("engineered_features", {}),
        )
        FEATURE_DRIFT_SCORE.set(score)
        if score > self.max_drift_score:
            logger.warning(
                "Обнаружен дрейф признаков: score=%.3f threshold=%.3f",
                score,
                self.max_drift_score,
            )
            if self.block_on_drift:
                self._fail("feature_drift", "Порог дрейфа признаков превышен")
        return score

    def _calculate_drift(self, frame: pd.DataFrame) -> float:
        """Рассчитать максимальный сдвиг среднего в единицах train standard deviation."""
        return self._normalized_mean_shift(frame, self.profile.get("raw_features", {}))

    @staticmethod
    def _normalized_mean_shift(
        frame: pd.DataFrame,
        reference: dict[str, Any],
    ) -> float:
        """Рассчитать максимальный нормированный сдвиг для общих колонок."""
        scores: list[float] = []
        for column, stats in reference.items():
            if column not in frame.columns:
                continue
            std = float(stats.get("std", 0.0))
            if std <= 0:
                continue
            current_mean = float(frame[column].mean())
            scores.append(abs(current_mean - float(stats["mean"])) / std)
        return max(scores, default=0.0)

    @staticmethod
    def _fail(reason: str, message: str, cause: Exception | None = None) -> None:
        """Учесть ошибку качества и выбросить безопасное исключение."""
        DATA_QUALITY_FAILURES.labels(reason).inc()
        if cause:
            raise DataQualityError(message) from cause
        raise DataQualityError(message)
