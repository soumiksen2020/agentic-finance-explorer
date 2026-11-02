
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from crewai.tools import tool


@tool("stock_price_analyzer")
def stock_price_analyzer(ticker: str):
    """
    Fetches historical stock data and calculates RSI
    and 20-day Moving Average.

    Accepts US tickers (NVDA, MSFT, AAPL) and Indian
    tickers (RELIANCE.NS, TCS.NS, RELIANCE.BO).
    """

    ticker = ticker.strip().upper()

    # Preserve an explicitly supplied exchange suffix.
    # Unqualified symbols are treated as US-listed tickers.
    yahoo_ticker = ticker

    data = yf.download(
        yahoo_ticker,
        period="1mo",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return f"Error: No data found for {yahoo_ticker}."

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data["RSI"] = ta.rsi(data["Close"], length=14)
    data["MA20"] = ta.sma(data["Close"], length=20)

    last_row = data.iloc[-1]

    current_price = float(last_row["Close"])
    rsi_val = last_row["RSI"]
    ma_val = last_row["MA20"]

    rsi_str = (
        f"{float(rsi_val):.2f}"
        if pd.notnull(rsi_val)
        else "Calculating..."
    )

    ma_str = (
        f"{float(ma_val):.2f}"
        if pd.notnull(ma_val)
        else "Calculating..."
    )

    return (
        f"--- Data for {yahoo_ticker} ---\n"
        f"Price: {current_price:.2f}\n"
        f"RSI: {rsi_str}\n"
        f"MA20: {ma_str}\n"
    )
