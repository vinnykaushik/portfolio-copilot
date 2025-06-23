from fastmcp import FastMCP
from pydantic import BaseModel
import yfinance as yf
from typing import Dict


class CustomerPortfolio(BaseModel):
    customer_id: str
    customer_name: str
    portfolio_value: int
    investments: list[dict]
    last_updated: str


mcp = FastMCP("mcp-portfolio-analytics")


def get_price_yahoo(symbol: str) -> float:
    """Fetches the latest price for a stock or crypto symbol from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info.get("regularMarketPrice", 0.0)
    except Exception:
        return 0.0


@mcp.tool()
def calculate_position_concentration_with_prices(
    portfolio: CustomerPortfolio, top_n: int = 5
) -> Dict:
    """
    Calculates HHI and Top-N position concentration using live prices.

    Args:
        portfolio (CustomerPortfolio): The portfolio data containing investments.
        top_n (int): The number of top positions to consider for concentration.
    """
    investments = portfolio.investments
    position_values = []

    for inv in investments:
        if inv["type"] in {"stocks", "crypto"}:
            symbol = inv["symbol"]
            yahoo_symbol = symbol if inv["type"] == "stocks" else f"{symbol}-USD"
            price = get_price_yahoo(yahoo_symbol)
            value = inv["quantity"] * price
            position_values.append({"name": symbol, "value": value})
        elif inv["type"] == "mutual_funds":
            name = inv["name"]
            value = inv["value"]
            position_values.append({"name": name, "value": value})

    portfolio_value = sum(p["value"] for p in position_values)

    positions = []
    for p in position_values:
        weight = p["value"] / portfolio_value if portfolio_value > 0 else 0.0
        positions.append({"name": p["name"], "weight": weight})

    hhi = sum(p["weight"] ** 2 for p in positions)
    sorted_positions = sorted(positions, key=lambda x: x["weight"], reverse=True)
    top_n_concentration = sum(p["weight"] for p in sorted_positions[:top_n])

    return {
        "portfolio_value": round(portfolio_value, 2),
        "hhi": round(hhi, 4),
        "top_n_concentration": round(top_n_concentration, 4),
        "positions": sorted_positions,
    }


if __name__ == "__main__":
    mcp.run()
