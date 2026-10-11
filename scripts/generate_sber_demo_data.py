"""Generate synthetic SBER demo data for testing."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_data(
    n_days: int = 250,
    start_price: float = 100.0,
    volatility: float = 0.02,
    output_path: Path = Path("data/sber/daily.csv"),
) -> None:
    """Generate synthetic OHLCV data for SBER.

    Args:
        n_days: Number of trading days to generate
        start_price: Starting price
        volatility: Daily volatility
        output_path: Path to save CSV file
    """
    np.random.seed(42)

    # Генерируем торговые даты без выходных
    dates = []
    current_date = datetime.now() - timedelta(days=n_days)
    for _ in range(n_days):
        # Пропускаем выходные
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)

    dates = dates[:n_days]
    dates.sort()

    # Генерируем цены через геометрическое броуновское движение
    returns = np.random.normal(0, volatility, len(dates))
    prices = start_price * np.cumprod(1 + returns)

    # Формируем OHLCV на основе цен
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Генерируем правдоподобные OHLC
        daily_range = close * volatility * np.random.uniform(0.5, 1.5)
        high = close + daily_range * np.random.uniform(0.3, 0.7)
        low = close - daily_range * np.random.uniform(0.3, 0.7)
        open_ = close + np.random.uniform(-daily_range * 0.3, daily_range * 0.3)

        # Соблюдаем ограничения OHLC
        high = max(high, open_, close)
        low = min(low, open_, close)

        # Генерируем объём
        volume = int(np.random.randint(1000000, 5000000))

        data.append({
            "timestamp": date.strftime("%Y-%m-%d"),
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume,
        })

    df = pd.DataFrame(data)

    # Сохраняем в CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} days of demo data")
    print(f"Saved to {output_path}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic SBER demo data")
    parser.add_argument(
        "--n-days",
        type=int,
        default=250,
        help="Number of trading days to generate (default: 250)",
    )
    parser.add_argument(
        "--start-price",
        type=float,
        default=100.0,
        help="Starting price (default: 100.0)",
    )
    parser.add_argument(
        "--volatility",
        type=float,
        default=0.02,
        help="Daily volatility (default: 0.02)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sber/daily.csv",
        help="Output file path (default: data/sber/daily.csv)",
    )

    args = parser.parse_args()

    generate_demo_data(
        n_days=args.n_days,
        start_price=args.start_price,
        volatility=args.volatility,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
