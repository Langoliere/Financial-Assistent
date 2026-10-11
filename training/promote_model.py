"""Проверка метрик и продвижение MLflow-модели из candidate в champion."""

import argparse
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

REQUIRED_METRICS = ("test_accuracy", "test_f1", "test_roc_auc")


def passes_quality_gates(
    candidate_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    min_accuracy: float,
    min_f1: float,
    min_roc_auc: float,
    max_regression: float,
) -> tuple[bool, list[str]]:
    """Проверить абсолютные пороги и допустимую деградацию относительно champion."""
    reasons: list[str] = []
    thresholds = {
        "test_accuracy": min_accuracy,
        "test_f1": min_f1,
        "test_roc_auc": min_roc_auc,
    }
    for metric, threshold in thresholds.items():
        value = candidate_metrics.get(metric)
        if value is None:
            reasons.append(f"отсутствует метрика {metric}")
        elif value < threshold:
            reasons.append(f"{metric}={value:.4f} ниже порога {threshold:.4f}")

    if champion_metrics:
        for metric in REQUIRED_METRICS:
            candidate = candidate_metrics.get(metric)
            champion = champion_metrics.get(metric)
            if candidate is not None and champion is not None:
                if candidate < champion - max_regression:
                    reasons.append(
                        f"{metric} ухудшилась более чем на {max_regression:.4f}: "
                        f"candidate={candidate:.4f}, champion={champion:.4f}"
                    )
    return not reasons, reasons


def promote(args: argparse.Namespace) -> None:
    """Продвинуть candidate в champion только после прохождения quality gates."""
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    candidate_version = client.get_model_version_by_alias(args.model_name, args.candidate_alias)
    candidate_run = client.get_run(candidate_version.run_id)
    candidate_metrics = candidate_run.data.metrics

    champion_metrics: dict[str, float] | None = None
    try:
        champion_version = client.get_model_version_by_alias(args.model_name, args.champion_alias)
        champion_metrics = client.get_run(champion_version.run_id).data.metrics
    except Exception:
        champion_version = None

    passed, reasons = passes_quality_gates(
        candidate_metrics,
        champion_metrics,
        args.min_accuracy,
        args.min_f1,
        args.min_roc_auc,
        args.max_regression,
    )
    if not passed:
        raise RuntimeError("Модель не прошла quality gates: " + "; ".join(reasons))

    if args.dry_run:
        print(f"Версия {candidate_version.version} прошла quality gates; dry-run завершён")
        return

    client.set_registered_model_alias(
        args.model_name,
        args.champion_alias,
        candidate_version.version,
    )
    client.set_model_version_tag(
        args.model_name,
        candidate_version.version,
        "promotion_status",
        "approved",
    )
    print(
        f"Версия {candidate_version.version} модели {args.model_name} "
        f"продвинута в alias {args.champion_alias}"
    )


def parse_args() -> argparse.Namespace:
    """Разобрать параметры CLI и значения quality gates из окружения."""
    default_store = (Path(__file__).parent.parent / "mlruns").as_uri()
    parser = argparse.ArgumentParser(description="Продвижение MLflow candidate в champion")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", default_store),
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", "sber_direction_lstm"),
    )
    parser.add_argument("--candidate-alias", default="candidate")
    parser.add_argument("--champion-alias", default="champion")
    parser.add_argument("--min-accuracy", type=float, default=0.50)
    parser.add_argument("--min-f1", type=float, default=0.45)
    parser.add_argument("--min-roc-auc", type=float, default=0.50)
    parser.add_argument("--max-regression", type=float, default=0.03)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    promote(parse_args())
