from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Annotated
import yfinance as yf


class Investment(BaseModel):
    """Individual investment holding in a portfolio"""

    type: Annotated[
        Literal["stocks", "crypto", "mutual_funds"],
        Field(description="Type of investment asset"),
    ]
    symbol: Annotated[
        str | None,
        Field(
            default=None, description="Ticker symbol (required for stocks and crypto)"
        ),
    ]
    quantity: Annotated[
        float | None,
        Field(
            default=None,
            ge=0,
            description="Number of units held (required for stocks or crypto)",
        ),
    ]
    name: Annotated[
        str | None,
        Field(
            default=None, description="Name of mutual fund (required for mutual funds)"
        ),
    ]
    value: Annotated[
        float | None,
        Field(
            default=None, ge=0, description="Estimated value of mutual fund investment"
        ),
    ]

    def model_post_init(self, __context):
        """Validate that required fields are present based on investment type"""
        if self.type in ["stocks", "crypto"]:
            if not self.symbol or self.quantity is None:
                raise ValueError(f"Symbol and quantity are required for {self.type}")
        elif self.type == "mutual_funds":
            if not self.name or self.value is None:
                raise ValueError("Name and value are required for mutual funds")


class CustomerPortfolio(BaseModel):
    """Customer portfolio containing investments and metadata"""

    customer_id: Annotated[str, Field(description="Unique identifier for the customer")]
    customer_name: Annotated[str, Field(description="Name of the customer")]
    investments: Annotated[
        List[Investment],
        Field(description="List of the customer's portfolio investments"),
    ]
    last_updated: Annotated[
        str, Field(description="Last update timestamp in ISO 8601 format")
    ]


class ConcentrationResult(BaseModel):
    """Results from portfolio concentration analysis"""

    portfolio_value: Annotated[
        float, Field(description="Total calculated market value of the portfolio")
    ]
    hhi: Annotated[
        float,
        Field(
            description="Herfindahl-Hirschman Index representing concentration (0-1)"
        ),
    ]
    top_n_concentration: Annotated[
        float, Field(description="Sum of weights of the top-N largest positions (0-1)")
    ]
    positions: Annotated[
        List[Dict[str, float | str]],
        Field(description="List of portfolio positions sorted by descending weight"),
    ]


class ExposureBreakdown(BaseModel):
    """Individual exposure category breakdown"""

    name: Annotated[str, Field(description="Category name")]
    weight: Annotated[float, Field(description="Percentage weight in portfolio (0-1)")]
    value: Annotated[float, Field(description="Dollar value of this category")]


class ExposureResult(BaseModel):
    """Results from portfolio exposure analysis"""

    total_portfolio_value: Annotated[
        float, Field(description="Total calculated market value of the portfolio")
    ]
    sector_exposure: Annotated[
        List[ExposureBreakdown], Field(description="Breakdown by sector classification")
    ]
    geography_exposure: Annotated[
        List[ExposureBreakdown],
        Field(description="Breakdown by company headquarters country"),
    ]
    market_cap_exposure: Annotated[
        List[ExposureBreakdown],
        Field(description="Breakdown by market capitalization buckets"),
    ]
    errors: Annotated[
        List[str], Field(description="List of any errors encountered during analysis")
    ]


mcp = FastMCP("mcp-portfolio-analytics")


@mcp.tool(
    name="calculate_position_concentration",
    description="Calculates position-level concentration metrics using live Yahoo Finance prices. Returns the portfolio value, Herfindahl-Hirschman Index (HHI), top-N concentration, and normalized position weights.",
)
def calculate_position_concentration(
    portfolio: Annotated[
        CustomerPortfolio,
        Field(
            description="Customer portfolio containing ID, name, and list of investments"
        ),
    ],
    top_n: Annotated[
        int,
        Field(
            default=5,
            ge=1,
            le=20,
            description="Number of top holdings to include in the concentration metric",
        ),
    ],
) -> ConcentrationResult:
    """
    Calculates HHI and Top-N position concentration using live prices.

    The Herfindahl-Hirschman Index (HHI) measures portfolio concentration:
    - Values closer to 0 indicate more diversification
    - Values closer to 1 indicate higher concentration
    - Calculated as sum of squared position weights

    Top-N concentration shows what percentage of the portfolio is held in the largest N positions.
    """
    investments = portfolio.investments
    position_values = []

    for inv in investments:
        if inv.type in {"stocks", "crypto"}:
            symbol = inv.symbol or ""
            price = get_price_yahoo(symbol)
            value = inv.quantity or 0.0 * price
            position_values.append({"name": symbol, "value": value})
        elif inv.type == "mutual_funds":
            name = inv.name
            value = inv.value
            position_values.append({"name": name, "value": value})

    portfolio_value = sum(p["value"] for p in position_values)

    positions = []
    for p in position_values:
        weight = p["value"] / portfolio_value if portfolio_value > 0 else 0.0
        positions.append({"name": p["name"], "weight": weight})

    hhi = sum(p["weight"] ** 2 for p in positions)
    sorted_positions = sorted(positions, key=lambda x: x["weight"], reverse=True)
    top_n_concentration = sum(p["weight"] for p in sorted_positions[:top_n])

    return ConcentrationResult(
        portfolio_value=round(portfolio_value, 2),
        hhi=round(hhi, 4),
        top_n_concentration=round(top_n_concentration, 4),
        positions=sorted_positions,
    )


def get_price_yahoo(symbol: str) -> float:
    """Fetches the latest price for a stock or crypto symbol from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info.get("regularMarketPrice", 0.0)
    except Exception:
        return 0.0


@mcp.tool(
    name="analyze_portfolio_exposures",
    description="Analyzes portfolio exposures across sector, geography, and market cap dimensions using live Yahoo Finance data. Returns percentage allocations and dollar values for each category.",
)
def analyze_portfolio_exposures(
    portfolio: Annotated[
        CustomerPortfolio,
        Field(
            description="Customer portfolio containing ID, name, and list of investments"
        ),
    ],
    min_threshold: Annotated[
        float,
        Field(
            default=0.01,
            ge=0.0,
            le=1.0,
            description="Minimum weight threshold to include in results (0.01 = 1%)",
        ),
    ],
) -> ExposureResult:
    """
    Analyzes portfolio exposures across multiple dimensions.

    Returns breakdown showing percentage allocation and dollar values for:
    - Sectors (using Yahoo Finance sector classification)
    - Geography (based on company headquarters)
    - Market cap buckets (Small <$2B, Mid $2B-$10B, Large >$10B)

    Only categories above the minimum threshold are included in results.
    """
    investments = portfolio.investments
    position_data = []
    errors = []

    # Collect position data with enriched information
    for inv in investments:
        try:
            if inv.type in {"stocks", "crypto"}:
                symbol = inv.symbol
                yahoo_symbol = symbol if inv.type == "stocks" else f"{symbol}-USD"

                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info

                price = info.get("regularMarketPrice", 0.0)
                value = inv.quantity * price

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
                        "type": inv.type,
                    }
                )

            elif inv.type == "mutual_funds":
                name = inv.name
                value = inv.value

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
            symbol = inv.symbol if inv.symbol else inv.name if inv.name else "Unknown"
            errors.append(f"Could not fetch data for {symbol}: {str(e)}")
            continue

    # Calculate total portfolio value
    total_value = sum(p["value"] for p in position_data)

    if total_value == 0:
        return ExposureResult(
            total_portfolio_value=0,
            sector_exposure=[],
            geography_exposure=[],
            market_cap_exposure=[],
            errors=errors + ["Portfolio has zero value"],
        )

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
                    ExposureBreakdown(
                        name=category,
                        weight=round(weight, 4),
                        value=round(data["value"], 2),
                    )
                )

        # Sort by weight descending
        return sorted(exposure_list, key=lambda x: x.weight, reverse=True)

    # Calculate exposures
    sector_exposure = aggregate_exposures(position_data, "sector")
    geography_exposure = aggregate_exposures(position_data, "country")
    market_cap_exposure = aggregate_exposures(position_data, "market_cap_category")

    return ExposureResult(
        total_portfolio_value=round(total_value, 2),
        sector_exposure=sector_exposure,
        geography_exposure=geography_exposure,
        market_cap_exposure=market_cap_exposure,
        errors=errors,
    )


if __name__ == "__main__":
    mcp.run()
