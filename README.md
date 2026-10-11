# InvestBot - Financial Direction Prediction Service

ML-powered service for predicting the direction of stock price movement on the next trading day.

## Overview

InvestBot uses an LSTM neural network to predict whether a stock's price will go up or down on the next trading day. The service is trained on historical OHLCV (Open, High, Low, Close, Volume) data from the Moscow Exchange (MOEX) for SBER shares.

### Features

- **Direction Prediction**: Binary classification (up/down) for next trading day
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages
- **Feature Engineering**: 33 features including calendar, dynamic, and technical indicators
- **REST API**: FastAPI-based service with health check and prediction endpoints
- **Docker Support**: Containerized deployment with docker-compose

## Project Structure

```
investbot/
├── src/investbot/
│   ├── main.py              # FastAPI application
│   ├── api/                 # API routes
│   ├── schemas/             # Pydantic request/response schemas
│   ├── services/            # Business logic (prediction, data loading)
│   ├── models/              # Model loading and prediction
│   ├── features/            # Feature engineering pipeline
│   └── ui/                  # Web UI
├── training/
│   ├── prepare_data.py      # Download historical data from MOEX
│   ├── train_direction.py   # Train LSTM model with MLflow logging
│   └── evaluate.py          # Evaluate trained model
├── scripts/
│   └── generate_sber_demo_data.py  # Generate synthetic demo data
├── tests/                   # Unit and integration tests
├── data/sber/
│   └── daily.csv            # Historical SBER data (or demo data)
├── artifacts/               # Trained model artifacts (backward compatibility)
├── mlruns/                  # MLflow experiment tracking data
├── notebooks/               # Jupyter notebooks for research
├── Dockerfile
├── docker-compose.yml
├── .env.example             # Environment variables template
└── pyproject.toml
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -e .
```

### 2. Generate Demo Data (Optional)

If you don't have real SBER data, generate synthetic demo data:

```bash
python scripts/generate_sber_demo_data.py
```

This will create `data/sber/daily.csv` with 250 trading days of synthetic OHLCV data.

**Note**: This is synthetic data for demonstration purposes only. Do not use for actual trading.

### 3. Configure MLflow (Optional)

Set up MLflow for model tracking and deployment:

```bash
# Set MLflow tracking URI (optional, defaults to file://./mlruns)
export MLFLOW_TRACKING_URI=file:///E:/Pet_Projects/Devin/Financial Assistent/mlruns

# Or copy .env.example to .env and configure there
cp .env.example .env
# Edit .env with your MLflow settings
```

### 4. Train the Model (If Not Already Trained)

```bash
# Download real data from MOEX (optional)
python training/prepare_data.py

# Train the model
python training/train_direction.py
```

This will:
- Save the trained model to `artifacts/direction_model/` (for backward compatibility)
- Log the model, scaler, and metadata to MLflow
- Register the model in MLflow as `sber_direction_lstm`

**Note**: The training script uses MLflow for experiment tracking. All hyperparameters, metrics, and artifacts are logged automatically.

### 5. Run the Service

```bash
# Local development with MLflow
export MLFLOW_TRACKING_URI=file:///E:/Pet_Projects/Devin/Financial Assistent/mlruns
uvicorn investbot.main:app --host 0.0.0.0 --port 8000 --reload

# Or without MLflow (uses local artifacts)
uvicorn investbot.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open the UI

Open your browser and navigate to:

```
http://localhost:8000/ui
```

Click "Получить предсказание" to get a prediction for the next trading day.

**Note**: The UI now fetches live data from MOEX via apimoex automatically.

### Docker Deployment

```bash
# Build and start the service
docker compose up --build

# Run in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the service
docker compose down
```

Open the UI at `http://localhost:8000/ui`.

## Training the Model

```bash
python training/train_direction.py
```

This will:
- Load data from `data/sber_ohlcv.csv`
- Split chronologically: 80% train, 10% validation, 10% test
- Engineer features using FeaturePipeline
- Train LSTM model with early stopping
- Calculate optimal classification threshold
- Save artifacts to `artifacts/direction_model/` (backward compatibility)
- Log model, scaler, and metadata to MLflow
- Register model in MLflow as `sber_direction_lstm`

**MLflow Integration**:
- All hyperparameters are logged automatically
- Training metrics (loss, accuracy, F1, ROC-AUC) are tracked
- Model is registered for easy deployment
- Artifacts are stored in `mlruns/` directory

Training typically takes 5-15 minutes depending on hardware.

### Step 3: Evaluate the Model

Evaluate the trained model on test data:

```bash
python training/evaluate.py
```

This will:
- Load trained model and artifacts
- Evaluate on test set
- Print metrics and classification report

### Optional: Scheduled Training

For production, you may want to retrain the model periodically (e.g., weekly). This can be done via cron:

```bash
# Add to crontab for weekly training on Sundays at 2 AM
0 2 * * 0 cd /path/to/investbot && python training/train_direction.py
```

**Note**: The current implementation only supports manual training. Scheduled training requires additional infrastructure (orchestrator, monitoring, etc.).

## How It Works

### Prediction Flow via UI

When you click "Получить предсказание" in the UI:

1. The UI calls `POST /api/ui/predict`
2. The service fetches the latest historical data from MOEX via apimoex
3. It takes the last 21 observations
4. It forms a sequence of 20 timesteps (the last 20 observations)
5. The sequence is passed through the feature pipeline
6. The trained LSTM model predicts the direction (up/down)
7. The result is returned to the UI with:
   - direction: up or down
   - probability_up: probability of price increase
   - threshold: classification threshold
   - model_version: version of the model used

### Prediction Flow via API

The generic prediction endpoint supports two modes:

**Mode 1: Provide OHLCV data manually**
```bash
curl -X POST http://localhost:8000/v1/predict/direction \
  -H "Content-Type: application/json" \
  -d '{
    "security": "SBER",
    "data": [...],  # Your OHLCV data
    "fetch_from_moex": false
  }'
```

**Mode 2: Fetch data from MOEX automatically**
```bash
curl -X POST http://localhost:8000/v1/predict/direction \
  -H "Content-Type: application/json" \
  -d '{
    "security": "SBER",
    "fetch_from_moex": true
  }'
```

### Model Loading

The service supports two model loading strategies:

1. **MLflow (recommended)**: Load model from MLflow registry
   - Set `MLFLOW_TRACKING_URI` environment variable
   - Set `MLFLOW_MODEL_NAME` (default: `sber_direction_lstm`)
   - Optionally set `MLFLOW_RUN_ID` to load a specific run

2. **Local artifacts (fallback)**: Load from `artifacts/direction_model/`
   - Used for backward compatibility
   - Used when MLflow is not configured

### Data Requirements

The service fetches data from MOEX automatically via apimoex. No local data file is required for the UI prediction endpoint.

For manual data submission via the generic API, provide OHLCV data with:
- `date`: ISO datetime
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `value`: Trading value (close * volume)

The data must:
- Be sorted chronologically
- Have at least 21 observations
- Have no duplicate timestamps
- Contain only positive prices
- Satisfy OHLC constraints (high >= open/close, low <= open/close)

## Running the Service

The service loads the trained model at startup via FastAPI lifespan. The model must be trained first using the scripts in `training/`.

### Local Development

```bash
# Run with uvicorn
uvicorn investbot.main:app --host 0.0.0.0 --port 8000 --reload

# Or directly
python -m investbot.main
```

The API will be available at `http://localhost:8000`

**Note**: If the model artifacts are not found in `artifacts/direction_model/`, the service will start in degraded mode. The `/health` endpoint will return `status: "degraded"` and prediction endpoints will return 503.

### Docker Deployment

#### Build and Run with Docker Compose

```bash
# Build and start the service
docker compose up --build

# Run in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the service
docker compose down
```

The API will be available at `http://localhost:8000`

#### Manual Docker Build

```bash
# Build the image
docker build -t investbot:latest .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  investbot:latest
```

## API Documentation

Once the service is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### GET /health

Health check endpoint. Returns service status and model loading state.

**Response (model loaded):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "direction-lstm-v1"
}
```

**Response (model not loaded):**
```json
{
  "status": "degraded",
  "model_loaded": false,
  "model_version": null,
  "error": "Model not found: artifacts/direction_model not found"
}
```

#### GET /v1/model-info

Get information about the loaded model.

**Response:**
```json
{
  "model_name": "direction-lstm",
  "model_version": "direction-lstm-v1",
  "lookback": 20,
  "features": 33,
  "trained_until": "2026-10-04T10:00:00",
  "metrics": {
    "val_accuracy": 0.55,
    "test_accuracy": 0.53
  }
}
```

**Error (model not loaded):** 503 Service Unavailable

#### GET /ui

Serves the web UI page for SBER direction prediction.

**Response:** HTML page with interface

#### POST /api/ui/predict

Predict direction using live SBER data from MOEX. Automatically fetches the latest data and makes a prediction.

**Request:** No body required

**Response:**
```json
{
  "ticker": "SBER",
  "as_of": "2026-10-07",
  "forecast_horizon": "next_trading_day",
  "direction": "up",
  "probability_up": 0.64,
  "threshold": 0.5,
  "model_version": "direction-lstm-v1",
  "generated_at": "2026-10-07T18:00:00Z",
  "last_observations_count": 21
}
```

**Error (MOEX fetch failed):** 503 with error message

**Error (model not loaded):** 503 Service Unavailable

#### POST /v1/predict/direction

Predict direction of price movement for the next trading day.

**Request (with manual data):**
```json
{
  "security": "SBER",
  "data": [
    {
      "date": "2026-10-01T00:00:00",
      "open": 250.0,
      "high": 255.0,
      "low": 248.0,
      "close": 252.0,
      "volume": 1000000,
      "value": 252000000.0
    },
    {
      "date": "2026-10-02T00:00:00",
      "open": 252.0,
      "high": 258.0,
      "low": 250.0,
      "close": 255.0,
      "volume": 1200000,
      "value": 306000000.0
    }
    // ... minimum 21 observations
  ],
  "fetch_from_moex": false
}
```

**Request (fetch from MOEX):**
```json
{
  "security": "SBER",
  "fetch_from_moex": true
}
```

**Response:**
```json
{
  "ticker": "SBER",
  "as_of": "2026-10-04T12:00:00Z",
  "forecast_horizon": "next_trading_day",
  "direction": "up",
  "probability_up": 0.67,
  "threshold": 0.5,
  "model_version": "direction-lstm-v1",
  "generated_at": "2026-10-04T12:00:00Z",
  "latency_ms": 12.5
}
```

**Error (model not loaded):** 503 Service Unavailable

## Example curl Commands

### Health Check
```bash
curl http://localhost:8000/health
```

**Response (model loaded):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "direction-lstm-v1"
}
```

**Response (model not loaded):**
```json
{
  "status": "degraded",
  "model_loaded": false,
  "model_version": null,
  "error": "Model not found: artifacts/direction_model not found"
}
```

### Model Info
```bash
curl http://localhost:8000/v1/model-info
```

**Response:**
```json
{
  "model_name": "direction-lstm",
  "model_version": "direction-lstm-v1",
  "lookback": 20,
  "features": 33,
  "trained_until": "2026-10-04T10:00:00",
  "metrics": {
    "val_accuracy": 0.55,
    "test_accuracy": 0.53
  }
}
```

### Direction Prediction (with manual data)
```bash
curl -X POST http://localhost:8000/v1/predict/direction \
  -H "Content-Type: application/json" \
  -d '{
    "security": "SBER",
    "data": [
      {
        "date": "2026-10-01T00:00:00",
        "open": 250.0,
        "high": 255.0,
        "low": 248.0,
        "close": 252.0,
        "volume": 1000000,
        "value": 252000000.0
      },
      {
        "date": "2026-10-02T00:00:00",
        "open": 252.0,
        "high": 258.0,
        "low": 250.0,
        "close": 255.0,
        "volume": 1200000,
        "value": 306000000.0
      }
    ]
  }'
```

**Response:**
```json
{
  "ticker": "SBER",
  "as_of": "2026-10-04T12:00:00Z",
  "forecast_horizon": "next_trading_day",
  "direction": "up",
  "probability_up": 0.67,
  "threshold": 0.5,
  "model_version": "direction-lstm-v1",
  "generated_at": "2026-10-04T12:00:00Z",
  "latency_ms": 12.5
}
```

## Testing

### Run All Tests
```bash
pytest -q
```

### Run Specific Test File
```bash
pytest tests/test_features.py -q
pytest tests/test_api.py -q
```

### Run with Coverage
```bash
pytest --cov=src/investbot --cov-report=html
```

## Code Quality

### Linting
```bash
ruff check .
```

### Auto-fix Linting Issues
```bash
ruff check . --fix
```

## Model Details

### Architecture

- **Model Type**: LSTM (Long Short-Term Memory)
- **Layers**: 2 LSTM layers (64 → 32 units) with dropout
- **Output**: Sigmoid activation for binary classification
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam with learning rate 0.001

### Features

The model uses 33 features:

1. **Market Features (6)**: open, high, low, close, volume, value
2. **Calendar Features (10)**: year, month, day, dayofweek, is_monday, is_friday, is_last_weekday, days_to_weekend, days_to_month_end, quarter
3. **Dynamic Features (9)**: returns (1, 5, 20 days), moving averages (5, 20), std (5, 20), range, body, volume_change
4. **Technical Indicators (8)**: RSI(14), MACD, MACD signal, Bollinger Bands (upper, middle, lower)

### Training

- **Lookback Window**: 20 trading days
- **Split**: 80% train, 10% validation, 10% test (chronological)
- **Batch Size**: 32
- **Early Stopping**: Patience 10 epochs on validation loss
- **Threshold**: Optimized on validation set for maximum F1 score

## Production-механизмы

### Продвижение моделей

Обучение больше не переключает production-модель автоматически. Новая версия получает alias `candidate`:

```bash
python training/train_direction.py
```

Проверка без изменения alias:

```bash
python training/promote_model.py --dry-run
```

Продвижение прошедшей quality gates версии в `champion`:

```bash
python training/promote_model.py
```

По умолчанию проверяются минимальные значения accuracy, F1 и ROC-AUC, а также отсутствие деградации более чем на 0.03 относительно текущего `champion`. Пороги можно передать аргументами CLI.

### Health и метрики

```text
GET /health/live   — liveness процесса
GET /health/ready  — readiness модели, возвращает 503 при degraded-состоянии
GET /health        — совместимый сводный статус
GET /metrics       — Prometheus-метрики
```

Метрики включают HTTP latency, длительность MOEX-запросов и инференса, число предсказаний, ошибки качества данных, drift score и готовность модели.

### Безопасность

Поддерживаются:

- ограниченный список CORS origins;
- Trusted Host middleware;
- опциональный `X-API-Key` для `/v1/*`;
- ограничение размера тела запроса;
- rate limit prediction endpoints;
- request ID;
- CSP, `X-Frame-Options`, `X-Content-Type-Options` и `Referrer-Policy`.

Основные параметры находятся в `.env.example`.

### Качество данных и drift

Перед инференсом проверяются:

- обязательные OHLCV-колонки и числовые типы;
- сортировка и уникальность дат;
- NaN и бесконечные значения;
- положительные цены и ограничения OHLC;
- свежесть данных MOEX;
- сдвиг средних относительно train-профиля в единицах стандартного отклонения.

При обучении профиль сохраняется в `data_profile.json`. Через `BLOCK_ON_DATA_DRIFT=true` можно блокировать инференс при превышении `MAX_DATA_DRIFT_SCORE`.

## Limitations & Risks

- **Market Noise**: Financial markets are highly stochastic; predictions may not be accurate
- **Historical Bias**: Model trained on historical data; may not perform well in different market conditions
- **Single Security**: Currently trained only on SBER; other securities may require retraining
- **No Fundamental Analysis**: Model uses only price/volume data; ignores news, earnings, macro factors
- **Concept Drift**: Model performance may degrade over time; regular retraining recommended

## Configuration

### Environment Variables

The service can be configured via environment variables:

- `PYTHONUNBUFFERED`: Disable Python output buffering (default: 1 in Docker)
- `MLFLOW_TRACKING_URI`: MLflow backend URI; omit it to use local artifacts
- `MLFLOW_MODEL_NAME`: Registered model name (default: `sber_direction_lstm`)
- `MLFLOW_MODEL_ALIAS`: Model alias loaded by inference (default: `champion`)
- `MLFLOW_RUN_ID`: Optional immutable run ID that overrides the model alias

### Configuration Files

- `.env.example`: Example environment variables (do not commit actual .env)
- `pyproject.toml`: Project dependencies and metadata

## Troubleshooting

### Model Not Loaded (503 Error)

If the API returns 503, ensure:
1. Model artifacts exist in `artifacts/direction_model/`
2. All required files are present: `model.keras`, `scaler_X.pkl`, `feature_schema.json`, `threshold.json`, `metadata.json`
3. Run `python training/train_direction.py` to train the model

### Insufficient Data Error

Manual prediction requests require at least 40 OHLCV observations: rolling indicators need 20 observations of warm-up and the model needs a 20-step sequence. MOEX-backed prediction fetches 60 observations automatically.

### Docker Build Issues

If Docker build fails:
1. Ensure Docker is running
2. Check network connectivity for downloading dependencies
3. Try building without cache: `docker compose build --no-cache`

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines here]
