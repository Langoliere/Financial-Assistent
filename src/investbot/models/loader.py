"""Model loader for loading trained model and preprocessing artifacts."""

import json
import logging
from pathlib import Path

import mlflow
import mlflow.keras
import tensorflow as tf
from mlflow import MlflowClient

from investbot.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load a Keras model and its matching fitted feature pipeline."""

    def __init__(
        self,
        artifacts_dir: Path | None = None,
        mlflow_tracking_uri: str | None = None,
        mlflow_model_name: str = "sber_direction_lstm",
        mlflow_run_id: str | None = None,
        mlflow_model_alias: str = "champion",
    ) -> None:
        """Initialize the model loader."""
        self.artifacts_dir = artifacts_dir
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_run_id = mlflow_run_id
        self.mlflow_model_alias = mlflow_model_alias
        self.model: tf.keras.Model | None = None
        self.pipeline: FeaturePipeline | None = None
        self.threshold = 0.5
        self.metadata: dict = {}
        self.data_profile: dict = {}

    def load(self) -> None:
        """Load from MLflow when configured, otherwise use local artifacts."""
        if self.mlflow_tracking_uri:
            self._load_from_mlflow()
            logger.info("Model and preprocessing pipeline loaded from MLflow")
            return
        if self.artifacts_dir is None:
            raise RuntimeError("No model source is configured")
        self._load_from_local(self.artifacts_dir)
        logger.info("Model and preprocessing pipeline loaded from local artifacts")

    def _load_from_mlflow(self) -> None:
        """Load a model and all fitted preprocessing artifacts from one MLflow run."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        client = MlflowClient()

        if self.mlflow_run_id:
            run_id = self.mlflow_run_id
            model_uri = f"runs:/{run_id}/model"
        else:
            model_version = client.get_model_version_by_alias(
                self.mlflow_model_name, self.mlflow_model_alias
            )
            run_id = model_version.run_id
            model_uri = f"models:/{self.mlflow_model_name}@{self.mlflow_model_alias}"

        self.model = mlflow.keras.load_model(model_uri)
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="artifacts",
        )
        self._load_pipeline_metadata(Path(downloaded))

    def _load_from_local(self, artifacts_dir: Path) -> None:
        """Load a model and fitted preprocessing artifacts from a local directory."""
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
        model_path = artifacts_dir / "model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self._load_pipeline_metadata(artifacts_dir)

    def _load_pipeline_metadata(self, artifacts_dir: Path) -> None:
        """Load fitted transforms, threshold, schema, and training metadata."""
        self.pipeline = FeaturePipeline.load_fitted_objects(artifacts_dir)

        threshold_path = artifacts_dir / "threshold.json"
        metadata_path = artifacts_dir / "metadata.json"
        if not threshold_path.exists():
            raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with threshold_path.open(encoding="utf-8") as file:
            self.threshold = float(json.load(file)["threshold"])
        with metadata_path.open(encoding="utf-8") as file:
            self.metadata = json.load(file)
        profile_path = artifacts_dir / "data_profile.json"
        if profile_path.exists():
            with profile_path.open(encoding="utf-8") as file:
                self.data_profile = json.load(file)

    def is_loaded(self) -> bool:
        """Return whether both the model and fitted pipeline are available."""
        return self.model is not None and self.pipeline is not None
