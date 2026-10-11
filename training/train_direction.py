"""Train LSTM model for direction prediction."""

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from investbot.features.pipeline import FeaturePipeline

# Фиксируем генераторы случайных чисел для воспроизводимости
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model(
    input_shape: tuple[int, int],
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    """Build LSTM model for direction prediction.

    Args:
        input_shape: Shape of input data (timesteps, features)
        lstm_units_1: Number of units in first LSTM layer
        lstm_units_2: Number of units in second LSTM layer
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = Sequential(
        [
            LSTM(lstm_units_1, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units_2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def calculate_threshold(y_val: np.ndarray, y_val_pred_proba: np.ndarray) -> float:
    """Calculate optimal threshold for binary classification.

    Uses ROC curve to find threshold that maximizes F1 score.

    Args:
        y_val: True validation labels
        y_val_pred_proba: Predicted probabilities on validation set

    Returns:
        Optimal threshold value
    """
    from sklearn.metrics import f1_score

    best_threshold = 0.5
    best_f1 = 0.0

    # Перебираем разные пороги классификации
    thresholds = np.arange(0.3, 0.7, 0.01)
    for threshold in thresholds:
        y_pred = (y_val_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold)


def evaluate_model(
    model: Sequential,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: Trained Keras model
        x_test: Test features
        y_test: Test labels
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred_proba = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
    }

    return metrics


def main():
    """Main training function."""
    # Настраиваем MLflow
    default_tracking_path = Path(__file__).parent.parent / "mlruns"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_path.as_uri())
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sber_direction_prediction")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "sber_direction_lstm")
    model_alias = os.getenv("MLFLOW_CANDIDATE_ALIAS", "candidate")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Запускаем MLflow run
    with mlflow.start_run():
        # Записываем гиперпараметры
        mlflow.log_params({
            "lstm_units_1": 64,
            "lstm_units_2": 32,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "lookback": 20,
            "target": "direction",
        })

        # Загружаем данные
        data_path = Path(__file__).parent.parent / "data" / "sber_ohlcv.csv"
        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            print("Please run prepare_data.py first to download the data.")
            return

        print("Loading data...")
        df = pd.read_csv(data_path, index_col="begin", parse_dates=True)
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        # Записываем сведения о данных
        mlflow.log_param("data_shape", str(df.shape))
        mlflow.log_param("data_start", str(df.index.min()))
        mlflow.log_param("data_end", str(df.index.max()))

        # Хронологическое разбиение: 80% train, 10% validation, 10% test
        n = len(df)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print("\nSplit sizes:")
        print(f"Train: {len(train_df)} ({len(train_df)/n:.1%})")
        print(f"Val: {len(val_df)} ({len(val_df)/n:.1%})")
        print(f"Test: {len(test_df)} ({len(test_df)/n:.1%})")

        # Инициализируем pipeline
        lookback = 20
        pipeline = FeaturePipeline(lookback=lookback)

        # Обучаем pipeline только на train-данных
        print("\nFitting pipeline on train data...")
        pipeline.fit(train_df)

        # Преобразуем validation и test с историческим контекстом
        print("Transforming data...")
        x_train, y_train = pipeline.transform(train_df)
        x_val, y_val = pipeline.transform(val_df, add_context=train_df)
        x_test, y_test = pipeline.transform(test_df, add_context=pd.concat([train_df, val_df]))

        print("\nSequence shapes:")
        print(f"Train: {x_train.shape}, {y_train.shape}")
        print(f"Val: {x_val.shape}, {y_val.shape}")
        print(f"Test: {x_test.shape}, {y_test.shape}")

        # Создаём модель
        input_shape = (lookback, len(FeaturePipeline.FEATURE_COLS))
        print(f"\nBuilding model with input shape: {input_shape}")
        model = build_model(
            input_shape=input_shape,
            lstm_units_1=64,
            lstm_units_2=32,
            dropout_rate=0.2,
            learning_rate=0.001,
        )
        model.summary()

        # Обучаем модель
        print("\nTraining model...")
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1,
        )

        # Записываем метрики обучения
        for epoch, loss in enumerate(history.history["loss"]):
            mlflow.log_metric("train_loss", loss, step=epoch)
        for epoch, val_loss in enumerate(history.history["val_loss"]):
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Рассчитываем оптимальный порог на validation-выборке
        print("\nCalculating optimal threshold...")
        y_val_pred_proba = model.predict(x_val, verbose=0).flatten()
        threshold = calculate_threshold(y_val, y_val_pred_proba)
        print(f"Optimal threshold: {threshold:.3f}")
        mlflow.log_param("threshold", threshold)

        # Оцениваем модель на test-выборке
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(model, x_test, y_test, threshold)
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

        # Оцениваем модель на validation-выборке
        val_metrics = evaluate_model(model, x_val, y_val, threshold)
        print(f"\nVal accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        # Записываем метрики в MLflow
        mlflow.log_metrics({
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
        })

        # Локально сохраняем артефакты для обратной совместимости
        artifacts_dir = Path(__file__).parent.parent / "artifacts" / "direction_model"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving artifacts to {artifacts_dir}...")

        # Сохраняем модель
        model_path = artifacts_dir / "model.keras"
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # Сохраняем обученные объекты pipeline
        pipeline.save_fitted_objects(artifacts_dir)
        print("Pipeline objects saved")

        # Сохраняем порог классификации
        threshold_path = artifacts_dir / "threshold.json"
        with open(threshold_path, "w") as f:
            json.dump({"threshold": threshold}, f, indent=2)
        print(f"Threshold saved to {threshold_path}")

        # Сохраняем профиль train-данных для контроля дрейфа при инференсе
        profile_columns = ["open", "high", "low", "close", "volume", "value"]
        train_features = pipeline._engineer_features(train_df).dropna(
            subset=FeaturePipeline.FEATURE_COLS
        )
        data_profile = {
            "raw_features": {
                column: {
                    "mean": float(train_df[column].mean()),
                    "std": float(train_df[column].std()),
                    "min": float(train_df[column].min()),
                    "max": float(train_df[column].max()),
                }
                for column in profile_columns
            },
            "engineered_features": {
                column: {
                    "mean": float(train_features[column].mean()),
                    "std": float(train_features[column].std()),
                }
                for column in FeaturePipeline.FEATURE_COLS
            },
            "observations": len(train_df),
            "created_at": pd.Timestamp.now().isoformat(),
        }
        with (artifacts_dir / "data_profile.json").open("w", encoding="utf-8") as file:
            json.dump(data_profile, file, indent=2)

        # Сохраняем метаданные обучения
        metadata: dict[str, Any] = {
            "model_version": "v1.0.0",
            "training_date": pd.Timestamp.now().isoformat(),
            "lookback_window": lookback,
            "feature_count": len(FeaturePipeline.FEATURE_COLS),
            "target": "direction",
            "train_period": {
                "start": train_df.index.min().isoformat(),
                "end": train_df.index.max().isoformat(),
            },
            "val_period": {
                "start": val_df.index.min().isoformat(),
                "end": val_df.index.max().isoformat(),
            },
            "test_period": {
                "start": test_df.index.min().isoformat(),
                "end": test_df.index.max().isoformat(),
            },
            "metrics": {
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
            },
            "hyperparameters": {
                "lstm_units_1": 64,
                "lstm_units_2": 32,
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": len(history.history["loss"]),
            },
            "threshold": threshold,
        }

        metadata_path = artifacts_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")

        # Записываем модель и preprocessing-артефакты в один MLflow run
        print("\nLogging artifacts to MLflow...")
        mlflow.log_artifacts(artifacts_dir, "artifacts")
        model_info = mlflow.keras.log_model(
            model,
            name="model",
            registered_model_name=model_name,
        )
        if model_info.registered_model_version:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(
                model_name,
                model_alias,
                model_info.registered_model_version,
            )

        print("\nTraining completed successfully!")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
