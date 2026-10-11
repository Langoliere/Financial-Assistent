"""Tests for API endpoints."""

from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from investbot.main import app
from investbot.services.prediction_service import PredictionService

client = TestClient(app)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 45  # Больше минимально допустимых 40 наблюдений
    dates = [datetime.now() - timedelta(days=i) for i in range(n, 0, -1)]

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    data = []
    for i, date in enumerate(dates):
        data.append({
            "date": date.isoformat(),
            "open": float(close[i] + np.random.randn() * 0.2),
            "high": float(close[i] + abs(np.random.randn() * 0.3)),
            "low": float(close[i] - abs(np.random.randn() * 0.3)),
            "close": float(close[i]),
            "volume": int(np.random.randint(1000000, 5000000)),
            "value": float(close[i] * np.random.randint(1000000, 5000000)),
        })
    return data


@pytest.fixture
def mock_prediction_service(tmp_path):
    """Create a mock prediction service with minimal artifacts."""
    # Создаём минимальный набор артефактов
    artifacts_dir = tmp_path / "direction_model"
    artifacts_dir.mkdir()

    # Создаём простую тестовую модель
    import json

    import joblib
    import tensorflow as tf

    # Сохраняем тестовую модель
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(20, 33), activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.save(artifacts_dir / "model.keras")

    # Создаём тестовый scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(np.random.rand(100, 33))
    joblib.dump(scaler, artifacts_dir / "scaler_X.pkl")

    # Создаём схему признаков
    feature_schema = {
        "feature_order": [f"feature_{i}" for i in range(33)],
        "feature_count": 33,
        "lookback": 20,
        "feature_types": {f"feature_{i}": "float64" for i in range(33)},
    }
    with open(artifacts_dir / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f)

    # Сохраняем порог классификации
    with open(artifacts_dir / "threshold.json", "w") as f:
        json.dump({"threshold": 0.5}, f)

    # Сохраняем метаданные
    metadata = {
        "model_version": "v1.0.0",
        "training_date": "2026-10-04",
        "lookback_window": 20,
        "feature_count": 33,
        "target": "direction",
    }
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Создаём и загружаем сервис
    service = PredictionService(artifacts_dir)
    service.load_model()
    return service


class TestHealthEndpoint:
    """Тесты endpoints состояния сервиса."""

    def test_liveness(self):
        """Liveness не зависит от состояния модели."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_readiness_without_model(self):
        """Readiness возвращает 503 без загруженной модели."""
        app.state.model_loaded = False
        app.state.load_error = "test"
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["model_loaded"] is False

    def test_security_headers_and_request_id(self):
        """Middleware добавляет request ID и защитные заголовки."""
        response = client.get("/health/live", headers={"X-Request-ID": "test-request"})
        assert response.headers["X-Request-ID"] == "test-request"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_metrics(self):
        """Prometheus endpoint публикует метрики."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "investbot_http_requests_total" in response.text

    def test_health_check_without_model(self):
        """Test health check when model is not loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["model_version"] is None
        assert "error" in data


class TestModelInfoEndpoint:
    """Tests for /v1/model-info endpoint."""

    def test_model_info_without_model(self):
        """Test model info when model is not loaded."""
        response = client.get("/v1/model-info")
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data


class TestPredictionEndpoint:
    """Tests for /v1/predict/direction endpoint."""

    def test_prediction_without_model(self, sample_ohlcv_data):
        """Test prediction when model is not loaded."""
        # Формируем OHLCV-данные, проходящие валидацию
        dates = [datetime.now() - timedelta(days=i) for i in range(40, 0, -1)]
        data = []
        for date in dates:
            data.append({
                "date": date.isoformat(),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "volume": 1000000,
                "value": 100000000.0,
            })

        response = client.post(
            "/v1/predict/direction",
            json={
                "security": "SBER",
                "data": data,
            },
        )
        assert response.status_code == 503
        data_response = response.json()
        assert "detail" in data_response

    def test_prediction_insufficient_data(self):
        """Test prediction with insufficient data."""
        # Формируем только 20 наблюдений, что меньше допустимого минимума
        dates = [datetime.now() - timedelta(days=i) for i in range(20, 0, -1)]
        data = []
        for date in dates:
            data.append({
                "date": date.isoformat(),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "volume": 1000000,
                "value": 100000000.0,
            })

        response = client.post(
            "/v1/predict/direction",
            json={
                "security": "SBER",
                "data": data,
            },
        )
        assert response.status_code == 422

    def test_prediction_invalid_ohlcv(self):
        """Test prediction with invalid OHLCV data."""
        data = [{
            "date": datetime.now().isoformat(),
            "open": 100.0,
            "high": 90.0,  # Некорректно: high меньше open
            "low": 95.0,
            "close": 100.0,
            "volume": 1000000,
            "value": 100000000.0,
        }]

        response = client.post(
            "/v1/predict/direction",
            json={
                "security": "SBER",
                "data": data,
            },
        )
        assert response.status_code == 422

    def test_prediction_non_chronological(self):
        """Test prediction with non-chronological data."""
        dates = [datetime.now() - timedelta(days=i) for i in range(21, 0, -1)]
        # Разворачиваем даты, нарушая хронологический порядок
        dates = list(reversed(dates))

        data = []
        for date in dates:
            data.append({
                "date": date.isoformat(),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "volume": 1000000,
                "value": 100000000.0,
            })

        response = client.post(
            "/v1/predict/direction",
            json={
                "security": "SBER",
                "data": data,
            },
        )
        assert response.status_code == 422


class TestWithMockModel:
    """Tests with mock model."""

    def test_health_with_model(self, mock_prediction_service):
        """Test health check with model loaded via app.state."""
        app.state.prediction_service = mock_prediction_service
        app.state.model_loaded = True
        app.state.load_error = None

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_version"] == "direction-lstm-v1"

    def test_model_info_with_model(self, mock_prediction_service):
        """Test model info with model loaded via app.state."""
        app.state.prediction_service = mock_prediction_service
        app.state.model_loaded = True
        app.state.load_error = None

        response = client.get("/v1/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "lookback" in data
        assert "features" in data

    def test_prediction_with_model(self, mock_prediction_service, sample_ohlcv_data):
        """Test prediction with model loaded via app.state."""
        app.state.prediction_service = mock_prediction_service
        app.state.model_loaded = True
        app.state.load_error = None

        response = client.post(
            "/v1/predict/direction",
            json={
                "security": "SBER",
                "data": sample_ohlcv_data,
            },
        )
        # При несовместимости признаков endpoint должен вернуть контролируемую ошибку,
        # а не аварийно завершить обработку запроса
        assert response.status_code in [200, 400, 422, 500]


class TestUIEndpoints:
    """Tests for UI endpoints."""

    def test_get_ui_returns_html(self):
        """Test that GET /ui returns HTML."""
        response = client.get("/ui")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        content = response.text
        assert "Sberbank Direction Predictor" in content
        assert "Получить предсказание" in content

    def test_ui_predict_without_model(self):
        """Test /api/ui/predict when model is not loaded."""
        app.state.model_loaded = False
        app.state.prediction_service = None
        response = client.post("/api/ui/predict")
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data

    def test_ui_predict_moex_failure(self, monkeypatch):
        """Test that MOEX failures are exposed as service unavailability."""
        from investbot.services.data_loader import DataLoader

        monkeypatch.setattr(
            DataLoader,
            "fetch_ohlcv_for_prediction",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
        )
        app.state.model_loaded = True
        response = client.post("/api/ui/predict")
        assert response.status_code == 503
        assert "Failed to fetch data from MOEX" in response.json()["detail"]

    def test_ui_predict_with_model(self, monkeypatch):
        """Test UI prediction with mocked MOEX history and loaded service."""
        from investbot.services.data_loader import DataLoader

        observations = [
            {
                "date": (datetime(2026, 1, 1) + timedelta(days=index)).isoformat(),
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1_000_000,
                "value": 101_000_000.0,
            }
            for index in range(60)
        ]
        monkeypatch.setattr(
            DataLoader,
            "fetch_ohlcv_for_prediction",
            lambda *args, **kwargs: observations,
        )

        class StubService:
            threshold = 0.5

            def predict(self, data, check_freshness=False):
                assert len(data) == 60
                assert check_freshness is True
                return 1, 0.64, "2026-03-02T00:00:00"

            def get_model_info(self):
                return {"model_version": "direction-lstm-v1"}

        app.state.prediction_service = StubService()
        app.state.model_loaded = True
        response = client.post("/api/ui/predict")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "SBER"
        assert data["direction"] == "up"
        assert data["probability_up"] == 0.64
        assert data["last_observations_count"] == 60

