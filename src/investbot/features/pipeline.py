"""Feature pipeline for financial time series prediction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from investbot.features.technical import bollinger_bands, rsi


class FeaturePipeline:
    """Pipeline for feature engineering and sequence generation.

    This pipeline:
    1. Engineers features from OHLCV data
    2. Scales features using MinMaxScaler
    3. Generates sequences for LSTM
    4. Saves all fitted transformers for inference
    """

    FEATURE_COLS = [
        "open",
        "high",
        "low",
        "close",
        "value",
        "volume",
        "year",
        "month",
        "day",
        "dayofweek",
        "is_monday",
        "is_friday",
        "is_last_weekday",
        "days_to_weekend",
        "days_to_month_end",
        "quarter",
        "is_earnings_season",
        "ret_1",
        "ret_5",
        "ret_20",
        "ma_5",
        "ma_20",
        "std_5",
        "std_20",
        "range",
        "body",
        "volume_change",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_high",
        "bb_mid",
        "bb_low",
    ]

    def __init__(self, lookback: int = 20):
        """Initialize feature pipeline.

        Args:
            lookback: Number of timesteps in sequence (default: 20)
        """
        self.lookback = lookback
        self.scaler_X: MinMaxScaler | None = None
        self.feature_schema: dict = {}
        self._is_fitted = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from OHLCV data.

        Only uses current and past observations. No future information.

        Args:
            df: DataFrame with OHLCV data, indexed by datetime

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Календарные признаки
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["day"] = df.index.day
        df["dayofweek"] = df.index.dayofweek

        df["is_monday"] = (df["dayofweek"] == 0).astype(int)
        df["is_friday"] = (df["dayofweek"] == 4).astype(int)
        df["is_last_weekday"] = (df["dayofweek"] >= 3).astype(int)
        df["days_to_weekend"] = 4 - df["dayofweek"]
        df["days_to_month_end"] = df.index.days_in_month - df.index.day
        df["quarter"] = df["month"].map(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
        )
        df["is_earnings_season"] = df["month"].isin([1, 4, 7, 10]).astype(int)

        # Динамические признаки
        df["ret_1"] = df["close"].pct_change(1)
        df["ret_5"] = df["close"].pct_change(5)
        df["ret_20"] = df["close"].pct_change(20)

        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["std_5"] = df["close"].rolling(5).std()
        df["std_20"] = df["close"].rolling(20).std()

        df["range"] = (df["high"] - df["low"]) / df["close"]
        df["body"] = (df["close"] - df["open"]) / df["open"]
        df["volume_change"] = df["volume"].pct_change(1)

        # Технические индикаторы
        df["rsi_14"] = rsi(df["close"], length=14)

        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        bb_high, bb_mid, bb_low = bollinger_bands(df["close"], length=20, std_dev=2.0)
        df["bb_high"] = bb_high
        df["bb_mid"] = bb_mid
        df["bb_low"] = bb_low

        return df

    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable for direction prediction.

        Target: 1 if next-day close > current close, else 0.
        Uses shift(-1) only for target, not for features.

        Args:
            df: DataFrame with close prices

        Returns:
            Series of target values
        """
        target = (df["close"].shift(-1) > df["close"]).astype(int)
        return target

    def _create_sequences(
        self, x: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Create sequences for LSTM.

        For each target position t, uses window [t-lookback+1, t].

        Args:
            x: Scaled feature array (n_samples, n_features)
            y: Optional target array (n_samples,)

        Returns:
            Tuple of (x_seq, y_seq) where:
            - x_seq: (n_sequences, lookback, n_features)
            - y_seq: (n_sequences,) or None if y is None
        """
        xs, ys = [], []

        for i in range(self.lookback - 1, len(x)):
            xs.append(x[i - self.lookback + 1 : i + 1])
            if y is not None:
                ys.append(y[i])

        x_seq = np.array(xs)
        y_seq = np.array(ys) if y is not None else None

        return x_seq, y_seq

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit the pipeline on training data.

        Args:
            df: DataFrame with OHLCV data, indexed by datetime

        Returns:
            self
        """
        # Формируем признаки
        df_features = self._engineer_features(df)

        # Удаляем строки с NaN, появившиеся из-за скользящих окон
        df_clean = df_features.dropna(subset=self.FEATURE_COLS)

        # Обучаем scaler только на признаках
        x = df_clean[self.FEATURE_COLS].values
        self.scaler_X = MinMaxScaler()
        self.scaler_X.fit(x)

        # Сохраняем схему признаков
        self.feature_schema = {
            "feature_order": self.FEATURE_COLS,
            "feature_count": len(self.FEATURE_COLS),
            "lookback": self.lookback,
            "feature_types": {
                col: str(df_clean[col].dtype) for col in self.FEATURE_COLS
            },
        }

        self._is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, add_context: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform data to sequences.

        Args:
            df: DataFrame with OHLCV data, indexed by datetime
            add_context: Optional DataFrame to prepend for context (e.g., train data for val)

        Returns:
            Tuple of (x_seq, y_seq) sequences
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        # Добавляем контекст, если он передан
        if add_context is not None:
            df_combined = pd.concat([add_context, df])
        else:
            df_combined = df

        # Формируем признаки
        df_features = self._engineer_features(df_combined)

        # Формируем целевую переменную
        y = self._create_target(df_features)

        # Удаляем строки с NaN
        df_clean = df_features.dropna(subset=self.FEATURE_COLS)
        y_clean = y.loc[df_clean.index]

        # Масштабируем признаки
        x = df_clean[self.FEATURE_COLS].values
        x_scaled = self.scaler_X.transform(x)

        # После добавления контекста оставляем только сегмент исходного DataFrame
        if add_context is not None:
            # Отдельно формируем признаки контекста для подсчёта валидных строк
            context_features = self._engineer_features(add_context)
            n_clean_context = len(context_features.dropna(subset=self.FEATURE_COLS))
            x_scaled = x_scaled[n_clean_context:]
            y_clean = y_clean.iloc[n_clean_context:]

        # Формируем последовательности
        x_seq, y_seq = self._create_sequences(x_scaled, y_clean.values)

        return x_seq, y_seq

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step.

        Args:
            df: DataFrame with OHLCV data, indexed by datetime

        Returns:
            Tuple of (x_seq, y_seq) sequences
        """
        self.fit(df)
        return self.transform(df)

    def save_fitted_objects(self, path: Path) -> None:
        """Save fitted transformers and schema.

        Args:
            path: Directory to save artifacts
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        path.mkdir(parents=True, exist_ok=True)

        # Сохраняем scaler
        import joblib

        joblib.dump(self.scaler_X, path / "scaler_X.pkl")

        # Сохраняем схему признаков
        with open(path / "feature_schema.json", "w") as f:
            json.dump(self.feature_schema, f, indent=2)

    @classmethod
    def load_fitted_objects(cls, path: Path, lookback: int = 20) -> "FeaturePipeline":
        """Load fitted transformers and schema.

        Args:
            path: Directory with saved artifacts
            lookback: Lookback window size

        Returns:
            Fitted FeaturePipeline instance
        """
        import joblib

        pipeline = cls(lookback=lookback)

        # Загружаем scaler
        pipeline.scaler_X = joblib.load(path / "scaler_X.pkl")

        # Загружаем схему признаков
        with open(path / "feature_schema.json") as f:
            pipeline.feature_schema = json.load(f)

        pipeline._is_fitted = True
        return pipeline
