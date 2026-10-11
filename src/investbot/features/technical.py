"""Technical indicators for financial time series."""

import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    Args:
        close: Series of closing prices
        length: RSI period (default: 14)

    Returns:
        Series of RSI values
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def ema(close: pd.Series, length: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    Args:
        close: Series of closing prices
        length: EMA period

    Returns:
        Series of EMA values
    """
    return close.ewm(span=length, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD indicator.

    Args:
        close: Series of closing prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        Tuple of (macd, signal_line)
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def bollinger_bands(close: pd.Series, length: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands.

    Args:
        close: Series of closing prices
        length: Moving average period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = close.rolling(window=length).mean()
    std = close.rolling(window=length).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower
