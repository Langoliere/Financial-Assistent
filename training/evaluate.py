"""Evaluate trained direction model on test data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from investbot.features.pipeline import FeaturePipeline


def load_model_and_artifacts(artifacts_dir: Path) -> tuple:
    """Load trained model and artifacts.

    Args:
        artifacts_dir: Path to artifacts directory

    Returns:
        Tuple of (model, pipeline, threshold, metadata)
    """
    # Загружаем модель
    model_path = artifacts_dir / "model.keras"
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Загружаем pipeline
    pipeline = FeaturePipeline.load_fitted_objects(artifacts_dir)
    print("Pipeline loaded")

    # Загружаем порог классификации
    threshold_path = artifacts_dir / "threshold.json"
    with open(threshold_path) as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]
    print(f"Threshold: {threshold:.3f}")

    # Загружаем метаданные
    metadata_path = artifacts_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    print("Metadata loaded")

    return model, pipeline, threshold, metadata


def evaluate_on_data(
    model: tf.keras.Model,
    pipeline: FeaturePipeline,
    df: pd.DataFrame,
    threshold: float,
    context_df: pd.DataFrame | None = None,
) -> dict:
    """Evaluate model on given data.

    Args:
        model: Trained Keras model
        pipeline: Fitted FeaturePipeline
        df: Data to evaluate
        threshold: Classification threshold
        context_df: Optional context data for sequence generation

    Returns:
        Dictionary of evaluation results
    """
    # Преобразуем данные
    x, y = pipeline.transform(df, add_context=context_df)

    # Выполняем инференс
    y_pred_proba = model.predict(x, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Рассчитываем метрики
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_pred_proba)),
    }

    # Формируем отчёт классификации
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    # Формируем матрицу ошибок
    cm = confusion_matrix(y, y_pred)

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred.tolist(),
        "probabilities": y_pred_proba.tolist(),
        "true_labels": y.tolist(),
    }


def main():
    """Main evaluation function."""
    artifacts_dir = Path(__file__).parent.parent / "artifacts" / "direction_model"

    if not artifacts_dir.exists():
        print(f"Artifacts directory not found: {artifacts_dir}")
        print("Please run train_direction.py first to train the model.")
        return

    # Загружаем модель и артефакты
    print("Loading model and artifacts...")
    model, pipeline, threshold, metadata = load_model_and_artifacts(artifacts_dir)

    # Загружаем тестовые данные
    data_path = Path(__file__).parent.parent / "data" / "sber_ohlcv.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please run prepare_data.py first to download the data.")
        return

    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, index_col="begin", parse_dates=True)
    print(f"Data shape: {df.shape}")

    # Воспроизводим разбиение, использованное при обучении
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"\nEvaluating on test set ({len(test_df)} samples)...")
    test_results = evaluate_on_data(
        model, pipeline, test_df, threshold, context_df=pd.concat([train_df, val_df])
    )

    print("\n=== Test Set Results ===")
    print(f"Accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {test_results['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {test_results['metrics']['roc_auc']:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        test_results["true_labels"],
        test_results["predictions"],
        target_names=["Down", "Up"],
        zero_division=0,
    ))

    print("\nConfusion Matrix:")
    cm = np.array(test_results["confusion_matrix"])
    print(f"True Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives: {cm[1, 1]}")

    # Сравниваем с сохранёнными метаданными
    print("\n=== Comparison with Training Metadata ===")
    saved_test_acc = metadata["metrics"]["test_accuracy"]
    current_test_acc = test_results["metrics"]["accuracy"]
    print(f"Saved test accuracy: {saved_test_acc:.4f}")
    print(f"Current test accuracy: {current_test_acc:.4f}")
    print(f"Difference: {abs(saved_test_acc - current_test_acc):.4f}")

    if abs(saved_test_acc - current_test_acc) < 0.001:
        print("✓ Metrics match saved metadata")
    else:
        print("⚠ Metrics differ from saved metadata (expected due to randomness)")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
