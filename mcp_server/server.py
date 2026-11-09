import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Add project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import existing CrewAI stock analysis tool
from tools import stock_price_analyzer


mcp = FastMCP("agentic-finance-explorer")


@mcp.tool()
def get_stock_technical_analysis(ticker: str) -> str:
    """
    Fetch stock price, RSI (14), 20-day SMA,
    and historical data coverage.
    Supports US and Indian stock tickers.
    """
    return stock_price_analyzer.func(ticker)


if __name__ == "__main__":
    mcp.run(transport="stdio")