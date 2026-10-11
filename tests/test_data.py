"""Тесты загрузки, валидации и мониторинга качества данных."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from investbot.api.routes import MIN_OBSERVATIONS, SBER_DATA_PATH
from investbot.services.data_quality import DataQualityError, DataQualityMonitor


class TestDataLoading:
    """Тесты локального демонстрационного файла SBER."""

    def test_data_file_exists(self):
        """Проверить наличие файла, если демо-данные были сгенерированы."""
        if not SBER_DATA_PATH.exists():
            pytest.skip(
                f"Файл не найден: {SBER_DATA_PATH}. "
                "Запустите 'python scripts/generate_sber_demo_data.py'."
            )
        assert SBER_DATA_PATH.exists()

    def test_data_file_format(self):
        """Проверить обязательные колонки файла."""
        if not SBER_DATA_PATH.exists():
            pytest.skip(f"Файл не найден: {SBER_DATA_PATH}")
        frame = pd.read_csv(SBER_DATA_PATH)
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        for column in required:
            assert column in frame.columns, f"Отсутствует колонка: {column}"

    def test_data_sufficient_observations(self):
        """Проверить достаточную длину локальной истории."""
        if not SBER_DATA_PATH.exists():
            pytest.skip(f"Файл не найден: {SBER_DATA_PATH}")
        frame = pd.read_csv(SBER_DATA_PATH)
        assert len(frame) >= MIN_OBSERVATIONS

    def test_data_sorted_and_unique(self):
        """Проверить сортировку и уникальность дат."""
        if not SBER_DATA_PATH.exists():
            pytest.skip(f"Файл не найден: {SBER_DATA_PATH}")
        frame = pd.read_csv(SBER_DATA_PATH)
        timestamps = pd.to_datetime(frame["timestamp"])
        assert timestamps.is_monotonic_increasing
        assert not timestamps.duplicated().any()


@pytest.fixture
def valid_observations() -> list[dict]:
    """Создать корректную свежую OHLCV-историю."""
    start = datetime.now(UTC) - timedelta(days=59)
    return [
        {
            "date": (start + timedelta(days=index)).isoformat(),
            "open": 100.0 + index,
            "high": 102.0 + index,
            "low": 99.0 + index,
            "close": 101.0 + index,
            "volume": 1_000_000 + index,
            "value": (101.0 + index) * (1_000_000 + index),
        }
        for index in range(60)
    ]


class TestDataQualityMonitor:
    """Тесты production-проверок входной истории."""

    def test_valid_data(self, valid_observations):
        """Корректные данные проходят проверку."""
        score = DataQualityMonitor().validate(valid_observations, check_freshness=True)
        assert score == 0.0

    def test_duplicate_dates_rejected(self, valid_observations):
        """Повторяющиеся даты отклоняются."""
        valid_observations[-1]["date"] = valid_observations[-2]["date"]
        with pytest.raises(DataQualityError, match="повторяющиеся"):
            DataQualityMonitor().validate(valid_observations)

    def test_invalid_ohlc_rejected(self, valid_observations):
        """Нарушение OHLC отклоняется."""
        valid_observations[-1]["high"] = 1.0
        with pytest.raises(DataQualityError, match="OHLC"):
            DataQualityMonitor().validate(valid_observations)

    def test_drift_score_uses_training_profile(self, valid_observations):
        """Дрейф рассчитывается относительно train-профиля."""
        profile = {
            "raw_features": {
                "close": {"mean": 100.0, "std": 10.0},
            }
        }
        score = DataQualityMonitor(profile).validate(valid_observations)
        assert score > 1.0

    def test_feature_drift_score(self):
        """Дрейф engineered features рассчитывается отдельно."""
        profile = {
            "engineered_features": {
                "ret_1": {"mean": 0.0, "std": 0.01},
            }
        }
        features = pd.DataFrame({"ret_1": [0.03, 0.04, 0.05]})
        score = DataQualityMonitor(profile).validate_feature_drift(features)
        assert score == pytest.approx(4.0)
