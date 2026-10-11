"""API routes for direction prediction service."""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from investbot.schemas.requests import DirectionPredictionRequest
from investbot.schemas.responses import (
    DirectionPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from investbot.services.data_loader import DataLoader
from investbot.services.data_quality import DataQualityError

logger = logging.getLogger(__name__)

router = APIRouter()

# Резервный путь к локальным данным SBER
SBER_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "sber" / "daily.csv"
MIN_OBSERVATIONS = 40
MOEX_OBSERVATIONS = 60


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Подтвердить, что процесс API принимает запросы."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(request: Request) -> Response:
    """Вернуть 200 только при готовой к инференсу модели."""
    if getattr(request.app.state, "model_loaded", False):
        return JSONResponse({"status": "ok", "model_loaded": True})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "degraded",
            "model_loaded": False,
            "error": getattr(request.app.state, "load_error", "Модель не загружена"),
        },
    )


@router.get("/metrics")
async def metrics() -> Response:
    """Опубликовать метрики приложения в формате Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint.

    Args:
        request: FastAPI request object

    Returns:
        Health status of the service
    """
    model_loaded = getattr(request.app.state, "model_loaded", False)
    load_error = getattr(request.app.state, "load_error", None)

    if model_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version="direction-lstm-v1",
        )
    else:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_version=None,
            error=load_error or "Model not loaded",
        )


@router.get("/v1/model-info", response_model=ModelInfoResponse)
async def get_model_info(request: Request) -> ModelInfoResponse:
    """Get model information.

    Args:
        request: FastAPI request object

    Returns:
        Model information

    Raises:
        HTTPException: If model is not loaded
    """
    if not getattr(request.app.state, "model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    service = request.app.state.prediction_service
    info = service.get_model_info()

    return ModelInfoResponse(
        model_name="direction-lstm",
        model_version=info.get("model_version", "direction-lstm-v1"),
        lookback=info.get("lookback_window", 20),
        features=info.get("feature_count", 33),
        trained_until=info.get("training_date", "unknown"),
        metrics=info.get("metrics", {}),
    )


@router.post(
    "/v1/predict/direction",
    response_model=DirectionPredictionResponse,
)
async def predict_direction(
    request: Request, req: DirectionPredictionRequest
) -> DirectionPredictionResponse:
    """Predict direction of price movement.

    Args:
        request: FastAPI request object
        req: Prediction request with OHLCV data or fetch_from_moex flag

    Returns:
        Direction prediction

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not getattr(request.app.state, "model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    start_time = time.time()

    try:
        service = request.app.state.prediction_service

        # Получаем данные MOEX, если это указано в запросе
        if req.fetch_from_moex:
            logger.info(f"Fetching latest data for {req.security} from MOEX")
            data_loader = DataLoader(security=req.security)
            data = data_loader.fetch_ohlcv_for_prediction(min_observations=MOEX_OBSERVATIONS)
            logger.info(f"Fetched {len(data)} observations from MOEX")
        elif req.data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either data or fetch_from_moex must be provided",
            )
        else:
            # Преобразуем Pydantic-модели в словари
            data = [item.model_dump() for item in req.data]

        # Выполняем предсказание
        prediction, probability, target_date_iso = service.predict(
            data,
            check_freshness=req.fetch_from_moex,
        )

        # Рассчитываем задержку
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Prediction completed in {latency_ms:.2f}ms")

        return DirectionPredictionResponse(
            ticker=req.security,
            as_of=datetime.now(UTC).isoformat(),
            forecast_horizon="next_trading_day",
            direction="up" if prediction == 1 else "down",
            probability_up=probability,
            threshold=service.threshold,
            model_version=service.get_model_info()["model_version"],
            generated_at=datetime.now(UTC).isoformat(),
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("Prediction dependency failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction dependency is unavailable",
        ) from e
    except (ValidationError, DataQualityError) as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction",
        )


@router.get("/ui")
async def get_ui() -> FileResponse:
    """Serve the UI HTML page.

    Returns:
        HTML page for the UI
    """
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if not ui_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="UI page not found",
        )
    return FileResponse(ui_path)


@router.post("/api/ui/predict")
async def ui_predict(request: Request) -> dict:
    """Predict direction using live SBER data from MOEX.

    This endpoint fetches the latest data from MOEX via apimoex,
    takes the last 21 observations, and uses the loaded model to predict direction.

    Args:
        request: FastAPI request object

    Returns:
        Prediction result as dictionary

    Raises:
        HTTPException: If model is not loaded, data fetching fails, or prediction fails
    """
    # Сначала проверяем готовность модели
    if not getattr(request.app.state, "model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    # Получаем данные с MOEX
    try:
        logger.info("Fetching latest SBER data from MOEX")
        data_loader = DataLoader(security="SBER")
        data = data_loader.fetch_ohlcv_for_prediction(min_observations=MOEX_OBSERVATIONS)
        logger.info(f"Fetched {len(data)} observations from MOEX")
    except RuntimeError as e:
        logger.error(f"Failed to fetch data from MOEX: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to fetch data from MOEX: {str(e)}",
        ) from e

    # Выполняем предсказание
    start_time = time.time()
    service = request.app.state.prediction_service
    try:
        prediction, probability, target_date_iso = service.predict(
            data,
            check_freshness=True,
        )
    except DataQualityError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    latency_ms = (time.time() - start_time) * 1000

    logger.info(f"Prediction completed in {latency_ms:.2f}ms")

    return {
        "ticker": "SBER",
        "as_of": data[-1]["date"],
        "forecast_horizon": "next_trading_day",
        "direction": "up" if prediction == 1 else "down",
        "probability_up": probability,
        "threshold": service.threshold,
        "model_version": service.get_model_info()["model_version"],
        "generated_at": datetime.now(UTC).isoformat(),
        "last_observations_count": len(data),
    }

