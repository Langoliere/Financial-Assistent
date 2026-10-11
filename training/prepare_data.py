"""Prepare historical OHLCV data from MOEX for training."""

import datetime
from pathlib import Path

import pandas as pd
import requests
from apimoex import get_market_candles


def fetch_sber_data(
    security: str = "SBER",
    interval: int = 24,
    years: int = 10,
    end_date: datetime.date | None = None,
) -> pd.DataFrame:
    """Fetch historical candles from MOEX.

    Args:
        security: Security ticker (default: SBER)
        interval: Candle interval in hours (default: 24 for daily)
        years: Number of years of historical data (default: 10)
        end_date: End date for data (default: today)

    Returns:
        DataFrame with OHLCV data indexed by datetime
    """
    if end_date is None:
        end_date = datetime.date.today()

    start_date = end_date - datetime.timedelta(days=360 * years)

    with requests.Session() as session:
        candles = get_market_candles(
            session,
            security=security,
            interval=interval,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            market="shares",
            engine="stock",
        )

    df = pd.DataFrame(candles)
    df["begin"] = pd.to_datetime(df["begin"])
    df.set_index("begin", inplace=True)

    # Сортируем по дате для сохранения хронологического порядка
    df = df.sort_index()

    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to CSV.

    Args:
        df: DataFrame to save
        output_path: Path to save CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")


def main():
    """Main function to prepare and save data."""
    # Получаем данные
    print("Fetching SBER data from MOEX...")
    df = fetch_sber_data(security="SBER", interval=24, years=10)

    # Сохраняем данные в локальный каталог
    data_dir = Path(__file__).parent.parent / "data"
    output_path = data_dir / "sber_ohlcv.csv"
    save_data(df, output_path)

    # Выводим сводную информацию
    print("\nData summary:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())


if __name__ == "__main__":
    main()
