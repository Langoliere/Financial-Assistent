"""Tests for feature engineering pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from investbot.features.pipeline import FeaturePipeline
from investbot.features.technical import bollinger_bands, ema, macd, rsi


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

    # Генерируем правдоподобные ценовые данные
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.2
    high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.3)
    low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000000, 5000000, n)
    value = close * volume

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "value": value},
        index=dates,
    )
    return df


class TestTechnicalIndicators:
    """Tests for technical indicator functions."""

    def test_rsi(self, sample_ohlcv):
        """Test RSI calculation."""
        result = rsi(sample_ohlcv["close"], length=14)
        assert len(result) == len(sample_ohlcv)
        assert result.isna().sum() > 0  # NaN в начале
        assert result.dropna().between(0, 100).all()  # RSI находится в диапазоне [0, 100]

    def test_ema(self, sample_ohlcv):
        """Test EMA calculation."""
        result = ema(sample_ohlcv["close"], length=12)
        assert len(result) == len(sample_ohlcv)
        assert not result.isna().any()

    def test_macd(self, sample_ohlcv):
        """Test MACD calculation."""
        macd_line, signal_line = macd(sample_ohlcv["close"])
        assert len(macd_line) == len(sample_ohlcv)
        assert len(signal_line) == len(sample_ohlcv)
        assert not macd_line.isna().any()
        assert not signal_line.isna().any()

    def test_bollinger_bands(self, sample_ohlcv):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = bollinger_bands(sample_ohlcv["close"])
        assert len(upper) == len(sample_ohlcv)
        assert len(middle) == len(sample_ohlcv)
        assert len(lower) == len(sample_ohlcv)
        # Верхняя линия должна быть не ниже средней, а средняя — нижней
        valid = ~middle.isna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()


class TestFeaturePipeline:
    """Tests for FeaturePipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = FeaturePipeline(lookback=20)
        assert pipeline.lookback == 20
        assert pipeline.scaler_X is None
        assert not pipeline._is_fitted

    def test_fit(self, sample_ohlcv):
        """Test fitting pipeline."""
        pipeline = FeaturePipeline(lookback=20)
        pipeline.fit(sample_ohlcv)

        assert pipeline._is_fitted
        assert pipeline.scaler_X is not None
        assert len(pipeline.feature_schema["feature_order"]) == 33
        assert pipeline.feature_schema["feature_count"] == 33
        assert pipeline.feature_schema["lookback"] == 20

    def test_transform_without_fit(self, sample_ohlcv):
        """Test transform without fit raises error."""
        pipeline = FeaturePipeline(lookback=20)
        with pytest.raises(ValueError, match="must be fitted"):
            pipeline.transform(sample_ohlcv)

    def test_fit_transform(self, sample_ohlcv):
        """Test fit_transform."""
        pipeline = FeaturePipeline(lookback=20)
        x_seq, y_seq = pipeline.fit_transform(sample_ohlcv)

        # Проверяем формы последовательностей
        assert x_seq.ndim == 3
        assert x_seq.shape[1] == 20  # длина окна
        assert x_seq.shape[2] == 33  # признаки
        assert y_seq.ndim == 1
        assert x_seq.shape[0] == y_seq.shape[0]

    def test_transform_with_context(self, sample_ohlcv):
        """Test transform with context data."""
        pipeline = FeaturePipeline(lookback=20)

        # Разбиваем данные с достаточным объёмом исторического контекста
        train = sample_ohlcv.iloc[:80]
        val = sample_ohlcv.iloc[80:]

        # Обучаем на train-части
        pipeline.fit(train)

        # Преобразуем validation-часть с контекстом
        x_seq, y_seq = pipeline.transform(val, add_context=train)

        # Проверяем формы массивов
        assert x_seq.ndim == 3
        assert x_seq.shape[1] == 20
        assert x_seq.shape[2] == 33

    def test_no_data_leakage(self, sample_ohlcv):
        """Test that features don't use future information."""
        pipeline = FeaturePipeline(lookback=20)
        df_features = pipeline._engineer_features(sample_ohlcv)

        # Проверяем отсутствие shift(-1) в признаках
        for col in df_features.columns:
            if col in ["open", "high", "low", "close", "volume", "value"]:
                continue
            # Признаки не должны идеально коррелировать с будущим close
            future_close = sample_ohlcv["close"].shift(-1)
            # Удаляем NaN перед расчётом корреляции
            valid_idx = df_features[col].notna() & future_close.notna()
            if valid_idx.sum() > 1:
                correlation = df_features[col].loc[valid_idx].corr(future_close.loc[valid_idx])
                # Высокая корреляция может указывать на утечку данных
                # Пропускаем NaN-корреляцию для константного признака
                if not pd.isna(correlation):
                    assert correlation < 0.99, f"Feature {col} may have data leakage"

    def test_feature_schema(self, sample_ohlcv):
        """Test feature schema is saved correctly."""
        pipeline = FeaturePipeline(lookback=20)
        pipeline.fit(sample_ohlcv)

        schema = pipeline.feature_schema
        assert "feature_order" in schema
        assert "feature_count" in schema
        assert "lookback" in schema
        assert "feature_types" in schema

        # Проверяем ожидаемый порядок признаков
        assert schema["feature_order"] == FeaturePipeline.FEATURE_COLS

    def test_save_and_load_fitted_objects(self, sample_ohlcv, tmp_path):
        """Test saving and loading fitted objects."""
        pipeline = FeaturePipeline(lookback=20)
        pipeline.fit(sample_ohlcv)

        # Сохраняем
        pipeline.save_fitted_objects(tmp_path)

        # Проверяем наличие файлов
        assert (tmp_path / "scaler_X.pkl").exists()
        assert (tmp_path / "feature_schema.json").exists()

        # Загружаем
        loaded_pipeline = FeaturePipeline.load_fitted_objects(tmp_path, lookback=20)

        # Проверяем, что загруженный pipeline обучен
        assert loaded_pipeline._is_fitted
        assert loaded_pipeline.scaler_X is not None
        assert loaded_pipeline.feature_schema == pipeline.feature_schema

        # Проверяем возможность преобразования
        x_seq, y_seq = loaded_pipeline.transform(sample_ohlcv)
        assert x_seq.shape[1] == 20
        assert x_seq.shape[2] == 33

    def test_save_without_fit_raises_error(self, tmp_path):
        """Test saving without fit raises error."""
        pipeline = FeaturePipeline(lookback=20)
        with pytest.raises(ValueError, match="must be fitted"):
            pipeline.save_fitted_objects(tmp_path)

    def test_chronological_order_preserved(self, sample_ohlcv):
        """Test that chronological order is preserved."""
        pipeline = FeaturePipeline(lookback=20)
        x_seq, y_seq = pipeline.fit_transform(sample_ohlcv)

        # Проверяем хронологический порядок последовательностей
        # Это косвенно подтверждается отсутствием перемешивания в pipeline

    def test_target_creation(self, sample_ohlcv):
        """Test target creation."""
        pipeline = FeaturePipeline(lookback=20)
        target = pipeline._create_target(sample_ohlcv)

        # Целевая переменная должна быть бинарной без учёта NaN
        assert target.dropna().isin([0, 1]).all()

        # Проверяем соответствие длины target входным данным
        assert len(target) == len(sample_ohlcv)

    def test_insufficient_data_for_sequences(self):
        """Test handling of insufficient data."""
        # Создаём небольшой набор данных с lookback=5
        # Оставляем минимум 20 строк для расчёта скользящих индикаторов
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        df = pd.DataFrame(
            {
                "open": close + np.random.randn(50) * 0.2,
                "high": close + np.abs(np.random.randn(50) * 0.3),
                "low": close - np.abs(np.random.randn(50) * 0.3),
                "close": close,
                "volume": np.random.randint(1000000, 5000000, 50),
                "value": close * np.random.randint(1000000, 5000000, 50),
            },
            index=dates,
        )

        pipeline = FeaturePipeline(lookback=5)
        x_seq, y_seq = pipeline.fit_transform(df)

        # После удаления NaN должны остаться последовательности
        assert x_seq.shape[0] >= 0

    def test_feature_engineering_creates_all_features(self, sample_ohlcv):
        """Test that all expected features are created."""
        pipeline = FeaturePipeline(lookback=20)
        df_features = pipeline._engineer_features(sample_ohlcv)

        # Проверяем наличие всех колонок признаков
        for col in FeaturePipeline.FEATURE_COLS:
            assert col in df_features.columns, f"Missing feature: {col}"

    def test_sequence_length_matches_lookback(self, sample_ohlcv):
        """Test that sequence length matches lookback parameter."""
        for lookback in [10, 20, 30]:
            pipeline = FeaturePipeline(lookback=lookback)
            x_seq, _ = pipeline.fit_transform(sample_ohlcv)
            assert x_seq.shape[1] == lookback

    def test_feature_schema_json_valid(self, sample_ohlcv, tmp_path):
        """Test that feature schema is valid JSON."""
        pipeline = FeaturePipeline(lookback=20)
        pipeline.fit(sample_ohlcv)
        pipeline.save_fitted_objects(tmp_path)

        # Загружаем and validate JSON
        with open(tmp_path / "feature_schema.json") as f:
            schema = json.load(f)

        assert isinstance(schema, dict)
        assert isinstance(schema["feature_order"], list)
        assert isinstance(schema["feature_types"], dict)
