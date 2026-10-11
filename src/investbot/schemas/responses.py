"""Response schemas for API."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok", "degraded"] = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_version: str | None = Field(None, description="Model version")
    error: str | None = Field(None, description="Error message if model not loaded")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    lookback: int = Field(..., description="Lookback window size")
    features: int = Field(..., description="Number of features")
    trained_until: str = Field(..., description="Training date")
    metrics: dict = Field(default_factory=dict, description="Model metrics")


class DirectionPredictionResponse(BaseModel):
    """Direction prediction response."""

    ticker: str = Field(..., description="Stock ticker")
    as_of: str = Field(..., description="Current date/time")
    forecast_horizon: str = Field(..., description="Forecast horizon")
    direction: Literal["up", "down"] = Field(..., description="Predicted direction")
    probability_up: float = Field(..., ge=0, le=1, description="Probability of up")
    threshold: float = Field(..., ge=0, le=1, description="Classification threshold")
    model_version: str = Field(..., description="Model version")
    generated_at: str = Field(..., description="Generation timestamp")
    latency_ms: float = Field(..., ge=0, description="Prediction latency in milliseconds")


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str = Field(..., description="Error message")

