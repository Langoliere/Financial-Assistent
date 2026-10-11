"""Predictor for making direction predictions."""

import logging
from typing import Literal

import numpy as np
import pandas as pd
import tensorflow as tf

from investbot.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class DirectionPredictor:
    """Predictor for direction prediction."""

    def __init__(
        self,
        model: tf.keras.Model,
        pipeline: FeaturePipeline,
        threshold: float,
    ):
        """Initialize predictor.

        Args:
            model: Trained Keras model
            pipeline: Fitted FeaturePipeline
            threshold: Classification threshold
        """
        self.model = model
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(
        self,
        data: list[dict],
    ) -> tuple[Literal[0, 1], float, pd.Timestamp]:
        """Make direction prediction.

        Args:
            data: List of OHLCV dictionaries

        Returns:
            Tuple of (prediction, probability, target_date)
        """
        # Преобразуем данные в DataFrame
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        # Обеспечиваем хронологический порядок
        df = df.sort_index()

        # Преобразуем всю историю, чтобы rolling-индикаторы получили контекст
        x_seq, _ = self.pipeline.transform(df)

        if x_seq.shape[0] == 0:
            raise ValueError("Insufficient data for prediction after NaN removal")

        # Используем последнюю последовательность
        x_last = x_seq[-1:]  # Форма: (1, lookback, features)

        # Выполняем инференс
        raw_prediction = self.model.predict(x_last, verbose=0)
        probability = float(np.asarray(raw_prediction).reshape(-1)[-1])
        prediction = 1 if probability >= self.threshold else 0

        # Приближённо определяем целевую дату после последнего наблюдения
        target_date = df.index[-1] + pd.Timedelta(days=1)

        logger.info(
            f"Prediction: {prediction}, Probability: {probability:.3f}, Target: {target_date}"
        )

        return prediction, probability, target_date

    def get_confidence(self, probability: float) -> Literal["low", "medium", "high"]:
        """Get confidence level from probability.

        Args:
            probability: Predicted probability

        Returns:
            Confidence level
        """
        if probability > 0.65 or probability < 0.35:
            return "high"
        elif probability > 0.55 or probability < 0.45:
            return "medium"
        else:
            return "low"
