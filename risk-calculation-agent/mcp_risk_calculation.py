from fastmcp import FastMCP
import mcp
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from abc import ABC, abstractmethod

mcp = FastMCP(name="Risk Calculation Agent")


# Your updated models (included for reference)
class Investment(BaseModel, ABC):
    """Base class for all investment holdings in a portfolio"""

    @abstractmethod
    def get_investment_type(self) -> str:
        """Return the type of investment"""
        pass


class StocksCrypto(Investment):
    """Investment model for stocks and cryptocurrency"""

    type: Annotated[
        str, Field(description="Type of investment asset (stocks or crypto)")
    ]
    symbol: Annotated[str, Field(description="Ticker symbol for the stock or crypto")]
    quantity: Annotated[float, Field(ge=0, description="Number of units held")]

    def get_investment_type(self) -> str:
        return self.type


class MutualFunds(Investment):
    """Investment model for mutual funds"""

    name: Annotated[str, Field(description="Name of the mutual fund")]
    value: Annotated[
        float, Field(ge=0, description="Estimated value of mutual fund investment")
    ]

    def get_investment_type(self) -> str:
        return "mutual_funds"


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


# CAPM Functions adapted for your new inheritance-based models


def get_risk_free_rate():
    """
    Get the current 10-year Treasury rate as risk-free rate
    Returns: float - risk-free rate as decimal
    """
    try:
        treasury = yf.Ticker("^TNX")
        hist = treasury.history(period="5d")
        risk_free_rate = hist["Close"][-1] / 100
        return risk_free_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return 0.04


def get_market_return(period="1y"):
    """
    Calculate market return using S&P 500 as proxy
    Args:
        period: str - time period for calculation
    Returns: float - market return as decimal
    """
    try:
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(period=period)

        if period == "1y":
            market_return = (hist["Close"][-1] / hist["Close"][0]) - 1
        else:
            days = len(hist)
            market_return = ((hist["Close"][-1] / hist["Close"][0]) ** (252 / days)) - 1

        return market_return
    except Exception as e:
        print(f"Error calculating market return: {e}")
        return 0.10


def get_stock_beta(symbol):
    """
    Get beta for a specific stock
    Args:
        symbol: str - stock ticker symbol
    Returns: float - beta value
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        beta = info.get("beta")

        if beta is None:
            raise ValueError(f"Beta not available for {symbol}")

        return beta
    except Exception as e:
        print(f"Error getting beta for {symbol}: {e}")
        return 1.0


def calculate_expected_return(symbol):
    """
    Calculate expected return using CAPM formula
    Args:
        symbol: str - stock ticker symbol
    Returns: dict - containing all CAPM components and expected return
    """
    rf = get_risk_free_rate()
    rm = get_market_return()
    beta = get_stock_beta(symbol)

    if beta is None:
        print(f"Could not determine beta for {symbol}")
        return None

    equity_risk_premium = rm - rf
    expected_return = rf + (beta * equity_risk_premium)

    return {
        "symbol": symbol,
        "risk_free_rate": rf,
        "market_return": rm,
        "beta": beta,
        "equity_risk_premium": equity_risk_premium,
        "expected_return": expected_return,
    }


def get_current_stock_price(symbol):
    """
    Get current stock price for calculating position values
    Args:
        symbol: str - stock ticker symbol
    Returns: float - current stock price
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        return hist["Close"][-1]
    except Exception as e:
        print(f"Error getting current price for {symbol}: {e}")
        return None


def calculate_investment_value(investment: Investment):
    """
    Calculate the current market value of an investment
    Args:
        investment: Investment - Base Investment instance (StocksCrypto or MutualFunds)
    Returns: float - current market value
    """
    investment_type = investment.get_investment_type()

    if investment_type == "mutual_funds":
        # MutualFunds has a direct value attribute
        mutual_fund = investment if isinstance(investment, MutualFunds) else None
        if mutual_fund is None:
            return 0
        return mutual_fund.value

    elif investment_type in ["stocks", "crypto"]:
        # StocksCrypto has symbol and quantity attributes
        stocks_crypto = investment if isinstance(investment, StocksCrypto) else None
        if stocks_crypto is None:
            # Type safety fallback - should not happen with proper data
            return 0
        current_price = get_current_stock_price(stocks_crypto.symbol)
        if current_price is None:
            return 0
        return stocks_crypto.quantity * current_price

    return 0


def calculate_portfolio_weights(portfolio: CustomerPortfolio):
    """
    Calculate weights for each investment in the portfolio
    Args:
        portfolio: CustomerPortfolio - Pydantic model instance
    Returns: dict - {symbol: weight} for stocks/crypto, excludes mutual funds
    """
    # Calculate total portfolio value
    total_value = sum(calculate_investment_value(inv) for inv in portfolio.investments)

    if total_value == 0:
        return {}

    weights = {}
    for investment in portfolio.investments:
        investment_type = investment.get_investment_type()
        if investment_type in ["stocks", "crypto"]:
            # Type checking - ensure it's a StocksCrypto instance
            if isinstance(investment, StocksCrypto):
                investment_value = calculate_investment_value(investment)
                weights[investment.symbol] = investment_value / total_value

    return weights


@mcp.tool()
def calculate_portfolio_expected_return(portfolio: CustomerPortfolio):
    """
    Calculate expected return for a CustomerPortfolio using CAPM
    Only includes stocks and crypto (excludes mutual funds)
    Args:
        portfolio: CustomerPortfolio - Pydantic model instance
    Returns: dict - portfolio analysis including expected return
    """
    # Get portfolio weights
    weights = calculate_portfolio_weights(portfolio)

    if not weights:
        return {
            "error": "No stocks or crypto found in portfolio for CAPM analysis",
            "portfolio_expected_return": None,
            "individual_investments": {},
        }

    portfolio_expected_return = 0
    investment_analysis = {}

    for symbol, weight in weights.items():
        stock_data = calculate_expected_return(symbol)
        if stock_data:
            investment_analysis[symbol] = {
                **stock_data,
                "weight": weight,
                "weighted_return": weight * stock_data["expected_return"],
            }
            portfolio_expected_return += weight * stock_data["expected_return"]

    return {
        "customer_id": portfolio.customer_id,
        "customer_name": portfolio.customer_name,
        "portfolio_expected_return": portfolio_expected_return,
        "total_portfolio_value": sum(
            calculate_investment_value(inv) for inv in portfolio.investments
        ),
        "analyzed_portion_value": sum(
            calculate_investment_value(inv)
            for inv in portfolio.investments
            if inv.get_investment_type() in ["stocks", "crypto"]
        ),
        "individual_investments": investment_analysis,
        "weights": weights,
    }


def get_portfolio_volatility_data(portfolio: CustomerPortfolio, period="1y"):
    """
    Get volatility data for portfolio VaR calculations
    Args:
        portfolio: CustomerPortfolio - Pydantic model instance
        period: str - time period for calculation
    Returns: dict - volatility data for each stock/crypto
    """
    volatility_data = {}

    for investment in portfolio.investments:
        investment_type = investment.get_investment_type()
        if investment_type in ["stocks", "crypto"] and isinstance(
            investment, StocksCrypto
        ):
            try:
                ticker = yf.Ticker(investment.symbol)
                hist = ticker.history(period=period)
                daily_returns = hist["Close"].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)  # Annualized

                volatility_data[investment.symbol] = {
                    "volatility": volatility,
                    "daily_returns": daily_returns.tolist(),
                }
            except Exception as e:
                print(f"Error calculating volatility for {investment.symbol}: {e}")
                volatility_data[investment.symbol] = {
                    "volatility": None,
                    "daily_returns": [],
                }

    return volatility_data


def get_portfolio_correlation_matrix(portfolio: CustomerPortfolio, period="1y"):
    """
    Calculate correlation matrix for stocks/crypto in the portfolio
    Args:
        portfolio: CustomerPortfolio - Pydantic model instance
        period: str - time period for calculation
    Returns: pandas.DataFrame - correlation matrix
    """
    symbols = []
    for investment in portfolio.investments:
        investment_type = investment.get_investment_type()
        if investment_type in ["stocks", "crypto"] and isinstance(
            investment, StocksCrypto
        ):
            symbols.append(investment.symbol)

    if len(symbols) < 2:
        print("Need at least 2 symbols for correlation matrix")
        return None

    try:
        raw_data = yf.download(symbols, period=period)
        if raw_data is None or raw_data.empty:
            print("No data returned from yfinance")
            return None

        data = raw_data["Close"]

        if isinstance(data, pd.Series):
            print(
                "Only one symbol returned valid data - cannot calculate correlation matrix"
            )
            return None

        returns = data.pct_change().dropna()
        correlation_matrix = returns.corr()
        return correlation_matrix
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None


def get_stocks_crypto_from_portfolio(
    portfolio: CustomerPortfolio,
) -> List[StocksCrypto]:
    """
    Extract only StocksCrypto investments from a portfolio
    Args:
        portfolio: CustomerPortfolio - Pydantic model instance
    Returns: List[StocksCrypto] - filtered list of stock/crypto investments
    """
    return [inv for inv in portfolio.investments if isinstance(inv, StocksCrypto)]


def get_mutual_funds_from_portfolio(portfolio: CustomerPortfolio) -> List[MutualFunds]:
    """
    Extract only MutualFunds investments from a portfolio
    Args:
        portfolio: CustomerPortfolio - Pydantic model instance
    Returns: List[MutualFunds] - filtered list of mutual fund investments
    """
    return [inv for inv in portfolio.investments if isinstance(inv, MutualFunds)]
