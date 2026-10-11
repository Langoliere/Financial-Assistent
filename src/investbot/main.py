"""Основное FastAPI-приложение сервиса прогнозирования InvestBot."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from investbot.api.routes import router
from investbot.observability import MODEL_READY, configure_logging
from investbot.security import ProductionMiddleware
from investbot.services.prediction_service import PredictionService

configure_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузить модель при старте и освободить ссылки при остановке."""
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_model_name = os.getenv("MLFLOW_MODEL_NAME", "sber_direction_lstm")
    mlflow_model_alias = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
    mlflow_run_id = os.getenv("MLFLOW_RUN_ID")
    artifacts_dir = Path(__file__).parent.parent.parent / "artifacts" / "direction_model"

    try:
        service = PredictionService(
            artifacts_dir=artifacts_dir,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_model_name=mlflow_model_name,
            mlflow_run_id=mlflow_run_id,
            mlflow_model_alias=mlflow_model_alias,
        )
        service.load_model()
        app.state.prediction_service = service
        app.state.model_loaded = True
        app.state.load_error = None
        MODEL_READY.set(1)
        logger.info("Модель и preprocessing pipeline успешно загружены")
    except Exception as error:
        logger.exception("Не удалось загрузить модель")
        app.state.model_loaded = False
        app.state.load_error = f"Ошибка загрузки модели: {error}"
        app.state.prediction_service = None
        MODEL_READY.set(0)

    yield

    logger.info("Остановка сервиса")
    app.state.prediction_service = None
    app.state.model_loaded = False
    MODEL_READY.set(0)


app = FastAPI(
    title="InvestBot Direction Prediction API",
    description="API прогнозирования направления цены SBER",
    version="1.1.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]
allowed_hosts = [
    host.strip()
    for host in os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1,testserver").split(",")
    if host.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
app.add_middleware(ProductionMiddleware)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "investbot.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
