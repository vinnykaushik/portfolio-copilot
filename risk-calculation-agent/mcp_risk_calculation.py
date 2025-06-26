from fastmcp import FastMCP
import mcp
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
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


def calculate_weights_from_investments(
    investments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate portfolio weights for stocks and crypto investments

    Args:
        investments: List of investment dictionaries

    Returns:
        Dictionary containing:
        - weights: Dict[symbol, weight] - portfolio weights
        - total_value: Total portfolio value including mutual funds
        - analyzed_value: Value of stocks/crypto only
        - symbols: List of symbols analyzed
        - excluded_investments: List of investments that couldn't be valued
    """
    try:
        # Calculate total portfolio value (including mutual funds)
        total_value = sum(calculate_investment_value(inv) for inv in investments)

        if total_value == 0:
            return {
                "error": "Portfolio has no value",
                "weights": {},
                "total_value": 0,
                "analyzed_value": 0,
                "symbols": [],
                "excluded_investments": [],
            }

        weights = {}
        analyzed_value = 0
        symbols = []
        excluded_investments = []

        # Calculate weights for stocks/crypto only
        for investment in investments:
            investment_type = investment.get("type", "").lower()
            if investment_type in ["stocks", "crypto"]:
                symbol = investment.get("symbol")
                if symbol:
                    investment_value = calculate_investment_value(investment)
                    if investment_value > 0:
                        weights[symbol] = investment_value / total_value
                        analyzed_value += investment_value
                        symbols.append(symbol)
                    else:
                        excluded_investments.append(
                            {
                                "symbol": symbol,
                                "type": investment_type,
                                "reason": "Could not determine current value",
                            }
                        )
                else:
                    excluded_investments.append(
                        {
                            "symbol": "N/A",
                            "type": investment_type,
                            "reason": "Missing symbol",
                        }
                    )

        return {
            "weights": weights,
            "total_value": total_value,
            "analyzed_value": analyzed_value,
            "symbols": symbols,
            "excluded_investments": excluded_investments,
        }

    except Exception as e:
        return {
            "error": f"Error calculating weights: {str(e)}",
            "weights": {},
            "total_value": 0,
            "analyzed_value": 0,
            "symbols": [],
            "excluded_investments": [],
        }


def calculate_portfolio_standard_deviation(
    weights: Dict[str, float], period: str = "1y"
) -> Dict[str, Any]:
    """
    Calculate portfolio standard deviation using weights and correlation matrix

    Args:
        weights: Dictionary of symbol -> weight mappings
        period: Time period for historical data

    Returns:
        Dictionary containing:
        - portfolio_std: Portfolio standard deviation (annualized)
        - individual_volatilities: Dict of symbol -> volatility
        - correlation_matrix: Correlation matrix as dict
        - excluded_symbols: List of symbols excluded due to data issues
        - effective_weights: Adjusted weights after excluding problematic symbols
        - warnings: List of warning messages
    """
    try:
        if not weights:
            return {
                "error": "No weights provided",
                "portfolio_std": None,
                "individual_volatilities": {},
                "correlation_matrix": {},
                "excluded_symbols": [],
                "effective_weights": {},
                "warnings": [],
            }

        symbols = list(weights.keys())
        individual_volatilities = {}
        excluded_symbols = []
        warnings = []

        # Get individual volatilities
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)

                if len(hist) < 30:  # Need at least 30 days for reliable volatility
                    excluded_symbols.append(symbol)
                    warnings.append(
                        f"Insufficient data for {symbol} (only {len(hist)} days)"
                    )
                    continue

                daily_returns = hist["Close"].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)  # Annualized

                if (
                    np.isnan(volatility) or volatility <= 0 or volatility > 5
                ):  # > 500% annual volatility seems unrealistic
                    excluded_symbols.append(symbol)
                    warnings.append(f"Invalid volatility for {symbol}: {volatility}")
                    continue

                individual_volatilities[symbol] = volatility

            except Exception as e:
                excluded_symbols.append(symbol)
                warnings.append(f"Error calculating volatility for {symbol}: {str(e)}")

        # Remove excluded symbols and recalculate weights
        effective_symbols = [s for s in symbols if s not in excluded_symbols]

        if not effective_symbols:
            return {
                "error": "No symbols with valid volatility data",
                "portfolio_std": None,
                "individual_volatilities": individual_volatilities,
                "correlation_matrix": {},
                "excluded_symbols": excluded_symbols,
                "effective_weights": {},
                "warnings": warnings,
            }

        # Recalculate weights for remaining symbols (normalize to sum to analyzed portion)
        total_effective_weight = sum(weights[s] for s in effective_symbols)
        effective_weights = {
            s: weights[s] / total_effective_weight for s in effective_symbols
        }

        # Handle single asset case
        if len(effective_symbols) == 1:
            symbol = effective_symbols[0]
            portfolio_std = individual_volatilities[symbol]
            return {
                "portfolio_std": portfolio_std,
                "individual_volatilities": individual_volatilities,
                "correlation_matrix": {symbol: {symbol: 1.0}},
                "excluded_symbols": excluded_symbols,
                "effective_weights": effective_weights,
                "warnings": warnings
                + ["Single asset portfolio - no correlation effects"],
            }

        # Get correlation matrix for multiple assets
        try:
            raw_data = yf.download(effective_symbols, period=period, progress=False)
            if raw_data is None or raw_data.empty:
                return {
                    "error": "Could not download correlation data",
                    "portfolio_std": None,
                    "individual_volatilities": individual_volatilities,
                    "correlation_matrix": {},
                    "excluded_symbols": excluded_symbols,
                    "effective_weights": effective_weights,
                    "warnings": warnings,
                }

            # Handle both single and multiple symbol cases
            if len(effective_symbols) == 1:
                correlation_matrix = {effective_symbols[0]: {effective_symbols[0]: 1.0}}
            else:
                data = raw_data["Close"]
                if isinstance(data, pd.Series):
                    correlation_matrix = {
                        effective_symbols[0]: {effective_symbols[0]: 1.0}
                    }
                else:
                    returns = data.pct_change().dropna()
                    corr_df = returns.corr()
                    correlation_matrix = corr_df.to_dict()

            # Calculate portfolio standard deviation using matrix formula
            # σ_p = √(w^T × Σ × w) where Σ is the covariance matrix
            portfolio_variance = 0

            for i, symbol_i in enumerate(effective_symbols):
                for j, symbol_j in enumerate(effective_symbols):
                    weight_i = effective_weights[symbol_i]
                    weight_j = effective_weights[symbol_j]
                    vol_i = individual_volatilities[symbol_i]
                    vol_j = individual_volatilities[symbol_j]
                    correlation = correlation_matrix.get(symbol_i, {}).get(symbol_j, 0)

                    portfolio_variance += (
                        weight_i * weight_j * vol_i * vol_j * correlation
                    )

            portfolio_std = np.sqrt(portfolio_variance)

            return {
                "portfolio_std": portfolio_std,
                "individual_volatilities": individual_volatilities,
                "correlation_matrix": correlation_matrix,
                "excluded_symbols": excluded_symbols,
                "effective_weights": effective_weights,
                "warnings": warnings,
            }

        except Exception as e:
            return {
                "error": f"Error calculating correlation matrix: {str(e)}",
                "portfolio_std": None,
                "individual_volatilities": individual_volatilities,
                "correlation_matrix": {},
                "excluded_symbols": excluded_symbols,
                "effective_weights": effective_weights,
                "warnings": warnings,
            }

    except Exception as e:
        return {
            "error": f"Error calculating portfolio standard deviation: {str(e)}",
            "portfolio_std": None,
            "individual_volatilities": {},
            "correlation_matrix": {},
            "excluded_symbols": [],
            "effective_weights": {},
            "warnings": [],
        }


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

        # Use the new weights function
        weights_result = calculate_weights_from_investments(investments)

        if "error" in weights_result:
            return {
                "error": weights_result["error"],
                "portfolio_expected_return": None,
                "individual_investments": {},
            }

        weights = weights_result["weights"]
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
            "total_portfolio_value": weights_result["total_value"],
            "analyzed_portion_value": weights_result["analyzed_value"],
            "individual_investments": investment_analysis,
            "weights": weights,
            "excluded_investments": weights_result["excluded_investments"],
        }

    except Exception as e:
        return {"error": f"Error calculating portfolio expected return: {str(e)}"}


def calculate_portfolio_var(
    portfolio_data: Dict[str, Any],
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    period: str = "1y",
) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR) for a portfolio using parametric method

    Args:
        portfolio_data: Dictionary containing portfolio information
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        time_horizon: Time horizon in days (default 1)
        period: Historical period for volatility calculation (default "1y")

    Returns:
        Dictionary containing VaR calculation results
    """
    try:
        investments = portfolio_data.get("investments", [])

        # Step 1: Calculate weights
        weights_result = calculate_weights_from_investments(investments)
        if "error" in weights_result:
            return {"error": weights_result["error"]}

        weights = weights_result["weights"]
        if not weights:
            return {"error": "No stocks or crypto found for VaR calculation"}

        # Step 2: Calculate expected return
        expected_return_result = calculate_portfolio_expected_return(portfolio_data)
        if "error" in expected_return_result:
            return {"error": expected_return_result["error"]}

        portfolio_expected_return = expected_return_result["portfolio_expected_return"]

        # Step 3: Calculate portfolio standard deviation
        std_result = calculate_portfolio_standard_deviation(weights, period)
        if "error" in std_result:
            return {"error": std_result["error"]}

        portfolio_std = std_result["portfolio_std"]
        if portfolio_std is None:
            return {"error": "Could not calculate portfolio standard deviation"}

        # Step 4: Convert confidence level to z-score
        z_score = norm.ppf(1 - confidence_level)  # Negative value for VaR

        # Step 5: Calculate VaR using parametric formula
        # VaR = μ + Z × σ, scaled for time horizon
        daily_var = portfolio_expected_return / 252 + z_score * portfolio_std / np.sqrt(
            252
        )
        var_absolute = (
            daily_var * np.sqrt(time_horizon) * weights_result["analyzed_value"]
        )
        var_percentage = daily_var * np.sqrt(time_horizon)

        return {
            "customer_id": portfolio_data.get("customer_id"),
            "customer_name": portfolio_data.get("customer_name"),
            "var_absolute": abs(var_absolute),  # Absolute dollar amount
            "var_percentage": abs(var_percentage),  # As percentage of portfolio
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon,
            "z_score": z_score,
            "portfolio_expected_return_annual": portfolio_expected_return,
            "portfolio_std_annual": portfolio_std,
            "total_portfolio_value": weights_result["total_value"],
            "analyzed_portion_value": weights_result["analyzed_value"],
            "coverage_percentage": (
                weights_result["analyzed_value"] / weights_result["total_value"]
                if weights_result["total_value"] > 0
                else 0
            ),
            "weights": weights,
            "excluded_investments": weights_result["excluded_investments"],
            "volatility_warnings": std_result["warnings"],
            "excluded_symbols": std_result["excluded_symbols"],
            "methodology": "Parametric VaR using historical volatility and correlation",
        }

    except Exception as e:
        return {"error": f"Error calculating portfolio VaR: {str(e)}"}


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
            raw_data = yf.download(symbols, period=period, progress=False)
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


def calculate_parametric_cvar(
    expected_return: float,
    portfolio_std: float,
    var_value: float,
    confidence_level: float,
    portfolio_value: float,
) -> Dict[str, Any]:
    """
    Calculate Conditional Value at Risk (CVaR) using parametric method

    Args:
        expected_return: Portfolio expected return (annualized)
        portfolio_std: Portfolio standard deviation (annualized)
        var_value: VaR value (as percentage)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        portfolio_value: Portfolio value for absolute CVaR calculation

    Returns:
        Dictionary containing CVaR calculation results
    """
    try:
        # Get z-score for the confidence level
        z_score = norm.ppf(1 - confidence_level)

        # Calculate probability density function at the z-score
        phi_z = norm.pdf(z_score)

        # Calculate tail probability (1 - confidence_level)
        tail_prob = 1 - confidence_level

        # Parametric CVaR formula: CVaR = μ + (φ(z) / (1-c)) × σ
        # This is the expected value of returns below the VaR threshold
        cvar_percentage = expected_return / 252 + (
            phi_z / tail_prob
        ) * portfolio_std / np.sqrt(252)
        cvar_absolute = abs(cvar_percentage * portfolio_value)

        return {
            "cvar_absolute": cvar_absolute,
            "cvar_percentage": abs(cvar_percentage),
            "z_score": z_score,
            "phi_z": phi_z,
            "tail_probability": tail_prob,
            "methodology": "Parametric CVaR assuming normal distribution",
        }

    except Exception as e:
        return {"error": f"Error calculating parametric CVaR: {str(e)}"}


def calculate_portfolio_cvar(
    portfolio_data: Dict[str, Any],
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    period: str = "1y",
) -> Dict[str, Any]:
    """
    Calculate Conditional Value at Risk (CVaR) for a portfolio

    Args:
        portfolio_data: Dictionary containing portfolio information
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        time_horizon: Time horizon in days (default 1)
        period: Historical period for volatility calculation (default "1y")

    Returns:
        Dictionary containing CVaR calculation results
    """
    try:
        # Step 1: Calculate VaR first (we'll reuse its components)
        var_result = calculate_portfolio_var(
            portfolio_data, confidence_level, time_horizon, period
        )

        if "error" in var_result:
            return {"error": var_result["error"]}

        # Step 2: Extract VaR components for CVaR calculation
        expected_return = var_result["portfolio_expected_return_annual"]
        portfolio_std = var_result["portfolio_std_annual"]
        var_percentage = var_result["var_percentage"]
        analyzed_value = var_result["analyzed_portion_value"]

        # Step 3: Calculate parametric CVaR
        cvar_result = calculate_parametric_cvar(
            expected_return=expected_return,
            portfolio_std=portfolio_std,
            var_value=var_percentage,
            confidence_level=confidence_level,
            portfolio_value=analyzed_value,
        )

        if "error" in cvar_result:
            return {"error": cvar_result["error"]}

        # Step 4: Scale for time horizon
        cvar_absolute_scaled = cvar_result["cvar_absolute"] * np.sqrt(time_horizon)
        cvar_percentage_scaled = cvar_result["cvar_percentage"] * np.sqrt(time_horizon)

        # Step 5: Combine results
        return {
            "customer_id": portfolio_data.get("customer_id"),
            "customer_name": portfolio_data.get("customer_name"),
            "cvar_absolute": cvar_absolute_scaled,  # Absolute dollar amount
            "cvar_percentage": cvar_percentage_scaled,  # As percentage of portfolio
            "var_absolute": var_result["var_absolute"],  # Include VaR for comparison
            "var_percentage": var_result["var_percentage"],
            "cvar_to_var_ratio": (
                cvar_percentage_scaled / var_result["var_percentage"]
                if var_result["var_percentage"] != 0
                else 0
            ),
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon,
            "z_score": cvar_result["z_score"],
            "phi_z": cvar_result["phi_z"],
            "tail_probability": cvar_result["tail_probability"],
            "portfolio_expected_return_annual": expected_return,
            "portfolio_std_annual": portfolio_std,
            "total_portfolio_value": var_result["total_portfolio_value"],
            "analyzed_portion_value": analyzed_value,
            "coverage_percentage": var_result["coverage_percentage"],
            "weights": var_result["weights"],
            "excluded_investments": var_result["excluded_investments"],
            "volatility_warnings": var_result["volatility_warnings"],
            "excluded_symbols": var_result["excluded_symbols"],
            "methodology": "Parametric CVaR using normal distribution assumption",
        }

    except Exception as e:
        return {"error": f"Error calculating portfolio CVaR: {str(e)}"}


# MCP Tool Definitions
mcp.tool(calculate_expected_return)
mcp.tool(calculate_portfolio_expected_return)
mcp.tool(calculate_portfolio_var)
mcp.tool(calculate_portfolio_cvar)


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
    
    var_result = calculate_portfolio_var(sample_portfolio)
    print("Portfolio VaR:", var_result)
 """
