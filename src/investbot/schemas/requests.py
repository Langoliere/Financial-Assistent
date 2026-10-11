"""Request schemas for API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class OHLCVData(BaseModel):
    """Single OHLCV observation."""

    date: datetime = Field(..., description="Date and time of the observation")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    value: float = Field(..., ge=0, description="Trading value in currency")

    @field_validator("high")
    @classmethod
    def high_must_be_ge_open(cls, v: float, info) -> float:
        """Validate that high is at least as high as open."""
        if "open" in info.data and v < info.data["open"]:
            raise ValueError("high must be >= open")
        return v

    @field_validator("high")
    @classmethod
    def high_must_be_ge_close(cls, v: float, info) -> float:
        """Validate that high is at least as high as close."""
        if "close" in info.data and v < info.data["close"]:
            raise ValueError("high must be >= close")
        return v

    @field_validator("low")
    @classmethod
    def low_must_be_le_open(cls, v: float, info) -> float:
        """Validate that low is at most as low as open."""
        if "open" in info.data and v > info.data["open"]:
            raise ValueError("low must be <= open")
        return v

    @field_validator("low")
    @classmethod
    def low_must_be_le_close(cls, v: float, info) -> float:
        """Validate that low is at most as low as close."""
        if "close" in info.data and v > info.data["close"]:
            raise ValueError("low must be <= close")
        return v


class DirectionPredictionRequest(BaseModel):
    """Request for direction prediction."""

    security: Literal["SBER"] = Field(..., description="Security ticker")
    data: list[OHLCVData] | None = Field(
        default=None,
        description="OHLCV data (minimum 21 observations). If None, fetches from MOEX.",
    )
    fetch_from_moex: bool = Field(
        default=False,
        description="If True, fetches latest data from MOEX instead of using provided data",
    )

    @field_validator("data")
    @classmethod
    def data_must_be_chronological(cls, v: list[OHLCVData] | None) -> list[OHLCVData] | None:
        """Validate that data is in chronological order."""
        if v is None:
            return v
        if len(v) < 40:
            raise ValueError("at least 40 observations are required for feature warm-up")
        dates = [item.date for item in v]
        if dates != sorted(dates):
            raise ValueError("data must be in chronological order")
        return v
