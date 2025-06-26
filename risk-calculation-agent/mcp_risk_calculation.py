from fastmcp import FastMCP
import mcp
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union

mcp = FastMCP(name="Risk Calculation Agent")


def get_risk_free_rate() -> float:
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


def get_market_return(period: str = "1y") -> float:
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


def get_stock_beta(symbol: str) -> float:
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


# @mcp.tool()
def calculate_expected_return(symbol: str) -> Dict[str, Any]:
    """
    Calculate expected return of a single stock using CAPM formula

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")

    Returns:
        Dictionary containing CAPM components and expected return
    """
    rf = get_risk_free_rate()
    rm = get_market_return()
    beta = get_stock_beta(symbol)

    if beta is None:
        return {"error": f"Could not determine beta for {symbol}"}

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


def get_current_stock_price(symbol: str) -> Union[float, None]:
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


def calculate_investment_value(investment: Dict[str, Any]) -> float:
    """
    Calculate the current market value of an investment
    Args:
        investment: Dictionary with investment details
    Returns: float - current market value
    """
    investment_type = investment.get("type", "").lower()

    if investment_type == "mutual_funds":
        return investment.get("value", 0)

    elif investment_type in ["stocks", "crypto"]:
        symbol = investment.get("symbol")
        quantity = investment.get("quantity", 0)

        if not symbol:
            return 0

        current_price = get_current_stock_price(symbol)
        if current_price is None:
            return 0
        return quantity * current_price

    return 0


@mcp.tool()
def calculate_portfolio_expected_return(
    portfolio_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate expected return for a customer portfolio using CAPM
    Only includes stocks and crypto (excludes mutual funds)

    Args:
        portfolio_data: Dictionary containing portfolio information with structure:
        {
            "customer_id": "string",
            "customer_name": "string",
            "investments": [
                {
                    "type": "stocks" | "crypto" | "mutual_funds",
                    "symbol": "string (for stocks/crypto)",
                    "quantity": number (for stocks/crypto),
                    "name": "string (for mutual funds)",
                    "value": number (for mutual funds)
                }
            ],
            "last_updated": "ISO 8601 timestamp"
        }

    Returns:
        Dictionary containing portfolio analysis including expected return
    """
    try:
        investments = portfolio_data.get("investments", [])

        # Calculate total portfolio value
        total_value = sum(calculate_investment_value(inv) for inv in investments)

        if total_value == 0:
            return {
                "error": "Portfolio has no value",
                "portfolio_expected_return": None,
                "individual_investments": {},
            }

        # Calculate weights for stocks/crypto only
        weights = {}
        analyzed_value = 0

        for investment in investments:
            investment_type = investment.get("type", "").lower()
            if investment_type in ["stocks", "crypto"]:
                symbol = investment.get("symbol")
                if symbol:
                    investment_value = calculate_investment_value(investment)
                    if investment_value > 0:
                        weights[symbol] = investment_value / total_value
                        analyzed_value += investment_value

        if not weights:
            return {
                "error": "No stocks or crypto found in portfolio for CAPM analysis",
                "portfolio_expected_return": None,
                "individual_investments": {},
            }

        # Calculate portfolio expected return
        portfolio_expected_return = 0
        investment_analysis = {}

        for symbol, weight in weights.items():
            stock_data = calculate_expected_return(symbol)
            if "error" not in stock_data:
                investment_analysis[symbol] = {
                    **stock_data,
                    "weight": weight,
                    "weighted_return": weight * stock_data["expected_return"],
                }
                portfolio_expected_return += weight * stock_data["expected_return"]

        return {
            "customer_id": portfolio_data.get("customer_id"),
            "customer_name": portfolio_data.get("customer_name"),
            "portfolio_expected_return": portfolio_expected_return,
            "total_portfolio_value": total_value,
            "analyzed_portion_value": analyzed_value,
            "individual_investments": investment_analysis,
            "weights": weights,
        }

    except Exception as e:
        return {"error": f"Error calculating portfolio expected return: {str(e)}"}


def get_portfolio_volatility_data(
    portfolio_data: Dict[str, Any], period: str = "1y"
) -> Dict[str, Any]:
    """
    Get volatility data for portfolio VaR calculations

    Args:
        portfolio_data: Dictionary containing portfolio information (same structure as calculate_portfolio_expected_return)
        period: Time period for calculation (default "1y")

    Returns:
        Dictionary with volatility data for each stock/crypto
    """
    try:
        investments = portfolio_data.get("investments", [])
        volatility_data = {}

        for investment in investments:
            investment_type = investment.get("type", "").lower()
            if investment_type in ["stocks", "crypto"]:
                symbol = investment.get("symbol")
                if not symbol:
                    continue

                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    daily_returns = hist["Close"].pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252)  # Annualized

                    volatility_data[symbol] = {
                        "volatility": volatility,
                        "daily_returns": daily_returns.tolist(),
                    }
                except Exception as e:
                    print(f"Error calculating volatility for {symbol}: {e}")
                    volatility_data[symbol] = {
                        "volatility": None,
                        "daily_returns": [],
                    }

        return volatility_data

    except Exception as e:
        return {"error": f"Error getting portfolio volatility data: {str(e)}"}


def get_portfolio_correlation_matrix(
    portfolio_data: Dict[str, Any], period: str = "1y"
) -> Dict[str, Any]:
    """
    Calculate correlation matrix for stocks/crypto in the portfolio

    Args:
        portfolio_data: Dictionary containing portfolio information (same structure as calculate_portfolio_expected_return)
        period: Time period for calculation (default "1y")

    Returns:
        Dictionary containing correlation matrix data
    """
    try:
        investments = portfolio_data.get("investments", [])
        symbols = []

        for investment in investments:
            investment_type = investment.get("type", "").lower()
            if investment_type in ["stocks", "crypto"]:
                symbol = investment.get("symbol")
                if symbol:
                    symbols.append(symbol)

        if len(symbols) < 2:
            return {"error": "Need at least 2 symbols for correlation matrix"}

        try:
            raw_data = yf.download(symbols, period=period)
            if raw_data is None or raw_data.empty:
                return {"error": "No data returned from yfinance"}

            data = raw_data["Close"]

            if isinstance(data, pd.Series):
                return {
                    "error": "Only one symbol returned valid data - cannot calculate correlation matrix"
                }

            returns = data.pct_change().dropna()
            correlation_matrix = returns.corr()

            # Convert to dictionary for JSON serialization
            correlation_dict = correlation_matrix.to_dict()

            return {"correlation_matrix": correlation_dict, "symbols": symbols}

        except Exception as e:
            return {"error": f"Error calculating correlation matrix: {str(e)}"}

    except Exception as e:
        return {"error": f"Error processing portfolio data: {str(e)}"}


def get_current_price(symbol: str) -> Dict[str, Any]:
    """
    Get current stock price for a given symbol

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")

    Returns:
        Dictionary containing current price information
    """
    try:
        price = get_current_stock_price(symbol)
        if price is None:
            return {"error": f"Could not fetch price for {symbol}"}

        return {"symbol": symbol, "current_price": price, "currency": "USD"}
    except Exception as e:
        return {"error": f"Error getting current price for {symbol}: {str(e)}"}


# Helper function to validate portfolio structure
def validate_portfolio_structure(portfolio_data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that portfolio data has the correct structure

    Args:
        portfolio_data: Portfolio dictionary to validate

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not isinstance(portfolio_data, dict):
        return False, "Portfolio data must be a dictionary"

    required_fields = ["customer_id", "customer_name", "investments"]
    for field in required_fields:
        if field not in portfolio_data:
            return False, f"Missing required field: {field}"

    investments = portfolio_data.get("investments", [])
    if not isinstance(investments, list):
        return False, "Investments must be a list"

    for i, investment in enumerate(investments):
        if not isinstance(investment, dict):
            return False, f"Investment {i} must be a dictionary"

        if "type" not in investment:
            return False, f"Investment {i} missing 'type' field"

        inv_type = investment.get("type", "").lower()
        if inv_type in ["stocks", "crypto"]:
            if "symbol" not in investment or "quantity" not in investment:
                return (
                    False,
                    f"Investment {i} of type '{inv_type}' missing 'symbol' or 'quantity'",
                )
        elif inv_type == "mutual_funds":
            if "name" not in investment or "value" not in investment:
                return (
                    False,
                    f"Investment {i} of type 'mutual_funds' missing 'name' or 'value'",
                )
        else:
            return False, f"Investment {i} has invalid type: {inv_type}"

    return True, ""


""" if __name__ == "__main__":
    # Example usage
    sample_portfolio = {
        "customer_id": "123",
        "customer_name": "John Doe",
        "investments": [
            {"type": "stocks", "symbol": "AAPL", "quantity": 10},
            {"type": "stocks", "symbol": "MSFT", "quantity": 5},
            {"type": "mutual_funds", "name": "Vanguard S&P 500", "value": 50000},
        ],
        "last_updated": "2025-06-25T10:00:00Z",
    }

    # Test the functions
    result = calculate_portfolio_expected_return(sample_portfolio)
    print("Portfolio Expected Return:", result)
 """
