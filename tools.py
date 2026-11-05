
import math
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from crewai.tools import tool


@tool("stock_price_analyzer")
def stock_price_analyzer(ticker: str) -> str:
    """
    Fetches historical stock data and calculates RSI (14)
    and 20-day Simple Moving Average.

    Supports US and Indian Yahoo Finance tickers.
    Examples: NVDA, MSFT, RELIANCE.NS, TCS.NS
    """

    ticker = ticker.strip().upper()

    if not ticker:
        return "Error: Please provide a valid stock ticker."

    try:
        data = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data.empty:
            return f"Error: No stock data found for {ticker}."

        # Handle yfinance MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:
            return f"Error: Closing price data unavailable for {ticker}."

        # Clean and validate closing prices
        close = pd.to_numeric(data["Close"], errors="coerce")
        close = close.dropna()

        if len(close) < 20:
            return (
                f"Error: Insufficient historical data for {ticker}. "
                f"Only {len(close)} valid closing prices found."
            )

        # Calculate technical indicators
        rsi = ta.rsi(close, length=14)
        ma20 = ta.sma(close, length=20)

        current_price = float(close.iloc[-1])
        rsi_value = rsi.iloc[-1]
        ma_value = ma20.iloc[-1]

        if not math.isfinite(current_price):
            return f"Error: Invalid current price for {ticker}."

        rsi_str = (
            f"{float(rsi_value):.2f}"
            if pd.notna(rsi_value) and math.isfinite(float(rsi_value))
            else "Unavailable"
        )

        ma_str = (
            f"{float(ma_value):.2f}"
            if pd.notna(ma_value) and math.isfinite(float(ma_value))
            else "Unavailable"
        )

        return (
            f"--- Data for {ticker} ---\n"
            f"Price: {current_price:.2f}\n"
            f"RSI (14): {rsi_str}\n"
            f"MA20: {ma_str}\n"
            f"Data points: {len(close)}"
        )

    except Exception as e:
        return f"Error fetching stock data for {ticker}: {str(e)}"
