"""MOEX historical market-data access for online inference."""

import logging
from datetime import date, timedelta

import pandas as pd
import requests
from apimoex import get_market_candles

from investbot.observability import MOEX_LATENCY, Timer

logger = logging.getLogger(__name__)


class DataLoader:
    """Fetch daily OHLCV observations from the MOEX ISS API through apimoex."""

    def __init__(self, security: str = "SBER") -> None:
        """Initialize a loader for one MOEX security."""
        self.security = security

    def fetch_latest_data(self, observations: int = 80) -> pd.DataFrame:
        """Fetch enough recent daily candles to build inference features."""
        end_date = date.today()
        start_date = end_date - timedelta(days=max(observations * 3, 180))
        logger.info(
            "Fetching MOEX candles security=%s start=%s end=%s",
            self.security,
            start_date,
            end_date,
        )
        timer = Timer()
        try:
            with requests.Session() as session:
                candles = get_market_candles(
                    session,
                    security=self.security,
                    interval=24,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    market="shares",
                    engine="stock",
                )
        except requests.RequestException as error:
            raise RuntimeError("Запрос рыночных данных MOEX завершился ошибкой") from error
        finally:
            MOEX_LATENCY.observe(timer.seconds)

        if not candles:
            raise RuntimeError(f"MOEX returned no daily candles for {self.security}")

        frame = pd.DataFrame(candles)
        required = {"begin", "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"MOEX response is missing columns: {sorted(missing)}")

        frame["begin"] = pd.to_datetime(frame["begin"], errors="raise")
        frame = frame.set_index("begin").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")].tail(observations)
        numeric = ["open", "high", "low", "close", "volume"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
        frame["value"] = frame.get("value", frame["close"] * frame["volume"])
        if frame[numeric].isna().any().any():
            raise RuntimeError("MOEX response contains missing OHLCV values")

        logger.info(
            "Fetched MOEX candles security=%s observations=%d as_of=%s",
            self.security,
            len(frame),
            frame.index[-1].date(),
        )
        return frame

    def fetch_ohlcv_for_prediction(self, min_observations: int = 60) -> list[dict]:
        """Return chronologically ordered OHLCV records for feature generation."""
        frame = self.fetch_latest_data(observations=max(min_observations, 60))
        if len(frame) < min_observations:
            raise RuntimeError(
                f"Insufficient MOEX history: {len(frame)} observations; "
                f"at least {min_observations} required"
            )
        return [
            {
                "date": timestamp.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "value": float(row["value"]),
            }
            for timestamp, row in frame.iterrows()
        ]
