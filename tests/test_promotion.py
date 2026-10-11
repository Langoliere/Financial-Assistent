"""Тесты quality gates для продвижения MLflow-моделей."""

from training.promote_model import passes_quality_gates


def test_candidate_passes_quality_gates() -> None:
    """Качественная candidate-модель проходит абсолютные и относительные пороги."""
    candidate = {"test_accuracy": 0.58, "test_f1": 0.55, "test_roc_auc": 0.61}
    champion = {"test_accuracy": 0.57, "test_f1": 0.54, "test_roc_auc": 0.60}
    passed, reasons = passes_quality_gates(candidate, champion, 0.5, 0.45, 0.5, 0.03)
    assert passed is True
    assert reasons == []


def test_candidate_with_regression_is_rejected() -> None:
    """Candidate с недопустимой деградацией не становится champion."""
    candidate = {"test_accuracy": 0.51, "test_f1": 0.46, "test_roc_auc": 0.51}
    champion = {"test_accuracy": 0.60, "test_f1": 0.58, "test_roc_auc": 0.62}
    passed, reasons = passes_quality_gates(candidate, champion, 0.5, 0.45, 0.5, 0.03)
    assert passed is False
    assert any("ухудшилась" in reason for reason in reasons)
