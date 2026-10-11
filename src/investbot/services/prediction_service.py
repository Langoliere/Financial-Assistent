"""Prediction service for direction prediction."""

import logging
from pathlib import Path

import pandas as pd

from investbot.models.loader import ModelLoader
from investbot.models.predictor import DirectionPredictor
from investbot.observability import (
    PREDICTION_LATENCY,
    PREDICTION_PROBABILITY,
    PREDICTIONS,
    Timer,
)
from investbot.services.data_quality import DataQualityMonitor

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making predictions."""

    def __init__(
        self,
        artifacts_dir: Path | None = None,
        mlflow_tracking_uri: str | None = None,
        mlflow_model_name: str = "sber_direction_lstm",
        mlflow_run_id: str | None = None,
        mlflow_model_alias: str = "champion",
    ):
        """Initialize prediction service.

        Args:
            artifacts_dir: Path to local artifacts directory (fallback)
            mlflow_tracking_uri: MLflow tracking URI
            mlflow_model_name: Name of the registered model in MLflow
            mlflow_run_id: Specific MLflow run ID to load from
            mlflow_model_alias: Registered model alias to load
        """
        self.artifacts_dir = artifacts_dir
        self.loader = ModelLoader(
            artifacts_dir=artifacts_dir,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_model_name=mlflow_model_name,
            mlflow_run_id=mlflow_run_id,
            mlflow_model_alias=mlflow_model_alias,
        )
        self.predictor: DirectionPredictor | None = None
        self.data_quality = DataQualityMonitor()

    def load_model(self) -> None:
        """Load model and artifacts."""
        if self.loader.mlflow_tracking_uri:
            logger.info("Loading model from MLflow")
        else:
            logger.info(f"Loading model from {self.artifacts_dir}")
        self.loader.load()
        self.predictor = DirectionPredictor(
            model=self.loader.model,
            pipeline=self.loader.pipeline,
            threshold=self.loader.threshold,
        )
        self.data_quality = DataQualityMonitor(self.loader.data_profile)
        logger.info("Model loaded successfully")

    def is_model_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.loader.is_loaded() and self.predictor is not None

    @property
    def threshold(self) -> float:
        """Get classification threshold.

        Returns:
            Classification threshold
        """
        return self.loader.threshold

    def predict(
        self,
        data: list[dict],
        check_freshness: bool = False,
    ) -> tuple[int, float, str]:
        """Проверить данные и выполнить предсказание направления."""
        if not self.is_model_loaded():
            raise RuntimeError("Модель не загружена")

        self.data_quality.validate(data, check_freshness=check_freshness)
        frame = pd.DataFrame(data)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        engineered = self.loader.pipeline._engineer_features(frame).dropna(
            subset=self.loader.pipeline.FEATURE_COLS
        )
        self.data_quality.validate_feature_drift(engineered)

        timer = Timer()
        prediction, probability, target_date = self.predictor.predict(data)
        PREDICTION_LATENCY.observe(timer.seconds)
        PREDICTION_PROBABILITY.set(probability)
        version = str(self.loader.metadata.get("model_version", "unknown"))
        direction = "up" if prediction == 1 else "down"
        PREDICTIONS.labels(direction, version).inc()
        return prediction, probability, target_date.isoformat()

    def get_model_info(self) -> dict:
        """Get model information.

        Returns:
            Dictionary with model information

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model is not loaded")

        metadata = self.loader.metadata
        return {
            "model_version": metadata.get("model_version", "unknown"),
            "target": metadata.get("target", "direction"),
            "lookback_window": metadata.get("lookback_window", 20),
            "feature_count": metadata.get("feature_count", 33),
            "training_date": metadata.get("training_date", "unknown"),
            "supported_securities": ["SBER"],
        }

