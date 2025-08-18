import os
import base64
from dotenv import load_dotenv

load_dotenv()

# Build Basic Auth header.
LANGFUSE_AUTH = base64.b64encode(
    f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
).decode()

# Configure OpenTelemetry endpoint & headers
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
    f"{os.environ.get('LANGFUSE_HOST')}/api/public/otel"
)
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

from langfuse import get_client

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

from google.adk.agents import Agent
from pydantic import BaseModel

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = "gemini-2.5-flash"


class CustomerPortfolio(BaseModel):
    customer_id: str
    customer_name: str
    portfolio_value: int
    investments: list[dict]
    last_updated: str


dummy_data = [
    {
        "customer_id": "12345",
        "customer_name": "Sundar Pichai",
        "portfolio_value": 100000,
        "investments": [
            {
                "type": "stocks",
                "symbol": "GOOGL",
                "quantity": 10,
            },
            {
                "type": "stocks",
                "symbol": "AAPL",
                "quantity": 5,
            },
            {
                "type": "mutual_funds",
                "name": "Vanguard 500 Index Fund",
                "value": 50000,
            },
            {
                "type": "stocks",
                "symbol": "AMZN",
                "quantity": 15,
            },
        ],
        "last_updated": "2025-05-01",
    },
    {
        "customer_id": "67890",
        "customer_name": "Satya Nadella",
        "portfolio_value": 250000,
        "investments": [
            {
                "type": "stocks",
                "symbol": "MSFT",
                "quantity": 25,
            },
            {
                "type": "stocks",
                "symbol": "NVDA",
                "quantity": 8,
            },
            {
                "type": "mutual_funds",
                "name": "Fidelity Technology Fund",
                "value": 75000,
            },
            {
                "type": "stocks",
                "symbol": "TSLA",
                "quantity": 12,
            },
            {
                "type": "bonds",
                "name": "US Treasury 10-Year",
                "value": 30000,
            },
        ],
        "last_updated": "2025-06-15",
    },
    {
        "customer_id": "11111",
        "customer_name": "Tim Cook",
        "portfolio_value": 180000,
        "investments": [
            {
                "type": "stocks",
                "symbol": "AAPL",
                "quantity": 50,
            },
            {
                "type": "stocks",
                "symbol": "META",
                "quantity": 7,
            },
            {
                "type": "mutual_funds",
                "name": "iShares S&P 500 ETF",
                "value": 65000,
            },
            {
                "type": "stocks",
                "symbol": "NFLX",
                "quantity": 3,
            },
        ],
        "last_updated": "2025-06-10",
    },
    {
        "customer_id": "22222",
        "customer_name": "Jeff Bezos",
        "portfolio_value": 500000,
        "investments": [
            {
                "type": "stocks",
                "symbol": "TSLA",
                "quantity": 100,
            },
            {
                "type": "stocks",
                "symbol": "SPACEX",
                "quantity": 20,
            },
            {
                "type": "crypto",
                "symbol": "BTC",
                "quantity": 2.5,
            },
            {
                "type": "mutual_funds",
                "name": "ARK Innovation ETF",
                "value": 80000,
            },
            {
                "type": "stocks",
                "symbol": "COIN",
                "quantity": 15,
            },
        ],
        "last_updated": "2025-06-20",
    },
    {
        "customer_id": "33333",
        "customer_name": "Carl Pei",
        "portfolio_value": 320000,
        "investments": [
            {
                "type": "stocks",
                "symbol": "NVDA",
                "quantity": 40,
            },
            {
                "type": "stocks",
                "symbol": "AMD",
                "quantity": 18,
            },
            {
                "type": "mutual_funds",
                "name": "Technology Select Sector SPDR Fund",
                "value": 90000,
            },
            {
                "type": "stocks",
                "symbol": "INTC",
                "quantity": 25,
            },
            {
                "type": "bonds",
                "name": "Corporate Bond Index Fund",
                "value": 45000,
            },
        ],
        "last_updated": "2025-06-18",
    },
]


# TOOLS ------------------------------------------------------------------------------------------------------------------------------------
def get_all_portfolios_tool() -> list[CustomerPortfolio]:
    """
    Fetches comprehensive portfolio data **for all customers** including all investment holdings and account information.

    Retrieves detailed portfolio information, including individual
    stock holdings with quantities, mutual fund investments with current values, bond positions,
    and cryptocurrency holdings where applicable.

    Returns:
        list[CustomerPortfolio]: A list of CustomerPortfolio objects, each containing:
            - customer_id (str): Unique identifier for the customer account
            - customer_name (str): Full name of the portfolio holder
            - portfolio_value (int): Total portfolio value in USD
            - investments (list[CustomerInvestment]): Detailed breakdown of all investment positions:
                * For stocks: type="stocks", symbol (str), quantity (int)
                * For cryptocurrency: type="crypto", symbol (str), quantity (float)
                * For mutual funds: type="mutual_funds", name (str), value (int)
                * For bonds: type="bonds", name (str), value (int)
            - last_updated (str): ISO date string of the last portfolio update

    Note:
        Currently returns mock data for demonstration purposes. In production, this would
        connect to actual financial data sources and customer management systems.
    """
    # Convert the raw dummy_data dictionaries to CustomerPortfolio objects
    return [CustomerPortfolio(**portfolio) for portfolio in dummy_data]


def get_portfolio_by_id_tool(customer_id: str) -> CustomerPortfolio | None:
    """
    Fetches comprehensive portfolio data for a **specific** customer.
    Retrieves detailed portfolio information, including individual stock holdings with quantities,
    mutual fund investments with current values, bond positions, and cryptocurrency holdings where applicable.

    Args:
        customer_id (str): Unique identifier for the customer account.

    Returns:
        CustomerPortfolio: A model containing the customer's portfolio data, structured as follows:
            - customer_id (str): Unique identifier for the customer account
            - customer_name (str): Full name of the portfolio holder
            - portfolio_value (int): Total portfolio value in USD
            - investments (list[dict]): Detailed breakdown of all investment positions:
                * For stocks: type="stocks", symbol (str), quantity (int)
                * For cryptocurrency: type="crypto", symbol (str), quantity (float)
                * For mutual funds: type="mutual_funds", name (str), value (int)
                * For bonds: type="bonds", name (str), value (int)
            - last_updated (str): ISO date string of the last portfolio update

    **Important**:
        If the customer ID does not match any existing portfolio, will return NONE.

    Note:
        Currently returns mock data for demonstration purposes. In production, this would
        connect to actual financial data sources and customer management systems.

    Example:
        The returned data structure follows this format:
        {
            "customer_id": "12345",
            "customer_name": "John Doe",
            "portfolio_value": 150000,
            "investments": [
                {"type": "stocks", "symbol": "AAPL", "quantity": 10},
                {"type": "mutual_funds", "name": "S&P 500 Fund", "value": 75000}
            ],
            "last_updated": "2025-06-23"
        }
    """
    for portfolio in dummy_data:
        if portfolio["customer_id"] == customer_id:
            return CustomerPortfolio(**portfolio)

    # If customer ID does not match, return None
    return None


def get_portfolio_by_name_tool(
    customer_name: str,
) -> CustomerPortfolio | list[CustomerPortfolio] | None:
    """
    Fetches comprehensive portfolio data for a **specific** customer by name. If there are multiple customers with the same name,
    it returns a list of portfolios for those customers.
    Retrieves detailed portfolio information, including individual stock holdings with quantities,
    mutual fund investments with current values, bond positions, and cryptocurrency holdings where applicable.

    Args:
        customer_name (str): Full name of the portfolio holder.

    Returns:
        list[CustomerPortfolio]: A list of models containing the customer's portfolio data, structured as follows:
            - customer_id (str): Unique identifier for the customer account
            - customer_name (str): Full name of the portfolio holder
            - portfolio_value (int): Total portfolio value in USD
            - investments (list[dict]): Detailed breakdown of all investment positions:
                * For stocks: type="stocks", symbol (str), quantity (int)
                * For cryptocurrency: type="crypto", symbol (str), quantity (float)
                * For mutual funds: type="mutual_funds", name (str), value (int)
                * For bonds: type="bonds", name (str), value (int)
            - last_updated (str): ISO date string of the last portfolio update

    **Important**:
        If the customer name does not match any existing portfolio, will return NONE.

    Note:
        Currently returns mock data for demonstration purposes. In production, this would
        connect to actual financial data sources and customer management systems.

    Example:
        The returned data structure follows this format:
        {
            "customer_id": "12345",
            "customer_name": "John Doe",
            "portfolio_value": 150000,
            "investments": [
                {"type": "stocks", "symbol": "AAPL", "quantity": 10},
                {"type": "mutual_funds", "name": "S&P 500 Fund", "value": 75000}
            ],
            "last_updated": "2025-06-23"
        }
    """
    matching_portfolios = [
        CustomerPortfolio(**portfolio)
        for portfolio in dummy_data
        if portfolio["customer_name"].lower() == customer_name.lower()
    ]
    if len(matching_portfolios) == 1:
        return matching_portfolios[0]
    return matching_portfolios


# AGENT ------------------------------------------------------------------------------------------------------------------------------------
customer_portfolio_prompt = f""""
  You are a specialized financial agent that retrieves and manages customer portfolio data from various investment accounts.
  Use the tools at your disposal to fetch portfolio information based on customer ID or name. 
  This agent provides comprehensive portfolio information including individual stock holdings, mutual fund investments, bond positions, and cryptocurrency assets.

  # Tools
  - `get_portfolio_by_id_tool`: Fetches customer portfolio data for a specific customer by their unique ID. 
                                   If the customer ID does not match any existing portfolio, it returns None.
                                   Output is structured as a CustomerPortfolio model.
  - `get_portfolio_by_name_tool`: Fetches portfolio data for a specific customer by their name. Returns a list of CustomerPortfolio models if multiple customers share the same name. You should display these to the user, then ask which one they want to proceed with.
  - `get_all_portfolios_tool`: Fetches comprehensive portfolio data for all customers. Output is structured as a list of CustomerPortfolio models.

  # Output Format
  Format the output as a JSON object with the same structure as the example below. 
  Note that stocks have a symbol and quantity, mutual funds have a name and value, and bonds have a name and value.
    {{
        "customer_id": "12345",
        "customer_name": "Sundar Pichai",
        "portfolio_value": 100000,
        "investments": [
            {{
                "type": "stocks",
                "symbol": "GOOGL",
                "quantity": 10
            }},
            {{
                "type": "stocks",
                "symbol": "AAPL",
                "quantity": 5
            }},
            {{
                "type": "mutual_funds",
                "name": "Vanguard 500 Index Fund",
                "value": 50000
            }},
            {{
                "type": "stocks",
                "symbol": "AMZN",
                "quantity": 15
            }}
        ],
        "last_updated": "2025-05-01"
    }},
"""

root_agent = Agent(
    name="customer_portfolio_agent",
    description="""An agent that fetches customer portfolio data. Formats this as a JSON object with the following structure:
    Note that stocks have a symbol and quantity, mutual funds have a name and value, and bonds have a name and value.
    {{
        "customer_id": "12345",
        "customer_name": "Sundar Pichai",
        "portfolio_value": 100000,
        "investments": [
            {{
                "type": "stocks",
                "symbol": "GOOGL",
                "quantity": 10
            }},
            {{
                "type": "stocks",
                "symbol": "AAPL",
                "quantity": 5
            }},
            {{
                "type": "mutual_funds",
                "name": "Vanguard 500 Index Fund",
                "value": 50000
            }},
            {{
                "type": "stocks",
                "symbol": "AMZN",
                "quantity": 15
            }}
        ],
        "last_updated": "2025-05-01"
    }},""",
    instruction=customer_portfolio_prompt,
    tools=[
        get_portfolio_by_id_tool,
        get_all_portfolios_tool,
        get_portfolio_by_name_tool,
    ],
    model=LLM_MODEL,
)
