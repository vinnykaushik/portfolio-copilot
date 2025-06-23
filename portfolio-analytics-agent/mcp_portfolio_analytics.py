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


@mcp.tool()
def calculate_position_concentration(
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


def get_price_yahoo(symbol: str) -> float:
    """Fetches the latest price for a stock or crypto symbol from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info.get("regularMarketPrice", 0.0)
    except Exception:
        return 0.0


@mcp.tool()
def analyze_portfolio_exposures(
    portfolio: CustomerPortfolio, min_threshold: float = 0.01
) -> Dict:
    """
    Analyzes portfolio exposures across sector, geography, and market cap dimensions.

    Args:
        portfolio (CustomerPortfolio): The portfolio data containing investments.
        min_threshold (float): Minimum weight threshold to include in results (default: 0.01 = 1%).

    Returns breakdown showing percentage allocation and dollar values for:
    - Sectors (using Yahoo Finance sector classification)
    - Geography (based on company headquarters)
    - Market cap buckets (Small <$2B, Mid $2B-$10B, Large >$10B)
    """
    investments = portfolio.investments
    position_data = []
    errors = []

    # Collect position data with enriched information
    for inv in investments:
        try:
            if inv["type"] in {"stocks", "crypto"}:
                symbol = inv["symbol"]
                yahoo_symbol = symbol if inv["type"] == "stocks" else f"{symbol}-USD"

                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info

                price = info.get("regularMarketPrice", 0.0)
                value = inv["quantity"] * price

                # Extract metadata with fallbacks
                sector = info.get("sector", "Unknown")
                country = info.get("country", "Unknown")
                market_cap = info.get("marketCap", 0)

                position_data.append(
                    {
                        "name": symbol,
                        "value": value,
                        "sector": sector,
                        "country": country,
                        "market_cap": market_cap,
                        "type": inv["type"],
                    }
                )

            elif inv["type"] == "mutual_funds":
                name = inv["name"]
                value = inv["value"]

                position_data.append(
                    {
                        "name": name,
                        "value": value,
                        "sector": "Mutual Funds",
                        "country": "Mixed",
                        "market_cap": 0,  # Not applicable
                        "type": "mutual_funds",
                    }
                )

        except Exception as e:
            symbol = inv.get("symbol", inv.get("name", "Unknown"))
            errors.append(f"Could not fetch data for {symbol}: {str(e)}")
            continue

    # Calculate total portfolio value
    total_value = sum(p["value"] for p in position_data)

    if total_value == 0:
        return {
            "total_portfolio_value": 0,
            "sector_exposure": [],
            "geography_exposure": [],
            "market_cap_exposure": [],
            "errors": errors + ["Portfolio has zero value"],
        }

    # Helper function to categorize market cap
    def categorize_market_cap(market_cap: float, investment_type: str) -> str:
        if investment_type == "mutual_funds":
            return "Mutual Funds"
        elif market_cap == 0:
            return "Unknown"
        elif market_cap < 2_000_000_000:
            return "Small Cap"
        elif market_cap < 10_000_000_000:
            return "Mid Cap"
        else:
            return "Large Cap"

    # Aggregate exposures
    def aggregate_exposures(positions, field_name):
        exposure_dict = {}
        for pos in positions:
            if field_name == "market_cap_category":
                category = categorize_market_cap(pos["market_cap"], pos["type"])
            else:
                category = pos[field_name]

            if category not in exposure_dict:
                exposure_dict[category] = {"value": 0, "weight": 0}

            exposure_dict[category]["value"] += pos["value"]

        # Calculate weights and convert to list
        exposure_list = []
        for category, data in exposure_dict.items():
            weight = data["value"] / total_value
            if weight >= min_threshold:  # Apply threshold filter
                exposure_list.append(
                    {
                        "name": category,
                        "weight": round(weight, 4),
                        "value": round(data["value"], 2),
                    }
                )

        # Sort by weight descending
        return sorted(exposure_list, key=lambda x: x["weight"], reverse=True)

    # Calculate exposures
    sector_exposure = aggregate_exposures(position_data, "sector")
    geography_exposure = aggregate_exposures(position_data, "country")
    market_cap_exposure = aggregate_exposures(position_data, "market_cap_category")

    return {
        "total_portfolio_value": round(total_value, 2),
        "sector_exposure": sector_exposure,
        "geography_exposure": geography_exposure,
        "market_cap_exposure": market_cap_exposure,
        "errors": errors,
    }


if __name__ == "__main__":
    mcp.run()
