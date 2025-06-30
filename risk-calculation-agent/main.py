#!/usr/bin/env python3
"""
Command Line Interface for Risk Calculation Agent

This CLI provides an interactive interface to communicate with the risk calculation agent.
"""

import asyncio
import sys
import json
from typing import Optional
from agent import get_response, extract_response_content


class RiskAgentCLI:
    def __init__(self):
        self.session_active = True

    def print_banner(self):
        """Print the CLI banner."""
        print("=" * 60)
        print("🔍 Risk Calculation Agent CLI")
        print("=" * 60)
        print("Welcome to the Risk Calculation Agent!")
        print("You can ask questions about portfolio risk analysis.")
        print("\nCommands:")
        print("  help    - Show this help message")
        print("  example - Show example portfolio data")
        print("  quit    - Exit the CLI")
        print("  exit    - Exit the CLI")
        print("-" * 60)

    def print_help(self):
        """Print help information."""
        print("\n📋 Help - Risk Calculation Agent")
        print("-" * 40)
        print("This agent can calculate various risk metrics for financial portfolios.")
        print("\nYou can ask questions like:")
        print("• What is the VaR of my portfolio?")
        print("• Calculate the conditional VaR for this portfolio")
        print("• Analyze the risk metrics for my investments")
        print("\nMake sure to provide portfolio data in JSON format when asked.")
        print("-" * 40)

    def print_example(self):
        """Print example portfolio data."""
        example_portfolio = {
            "customer_id": "12345",
            "customer_name": "John Doe",
            "portfolio_value": 250000,
            "investments": [
                {"type": "stocks", "symbol": "AAPL", "quantity": 100},
                {"type": "stocks", "symbol": "GOOGL", "quantity": 50},
                {
                    "type": "mutual_funds",
                    "name": "Vanguard S&P 500 ETF",
                    "value": 75000,
                },
            ],
            "last_updated": "2025-06-26",
        }

        print("\n📊 Example Portfolio Data:")
        print("-" * 30)
        print(json.dumps(example_portfolio, indent=2))
        print("-" * 30)

    def format_response(self, response: str) -> str:
        """Format the agent response for better readability."""
        # Simple formatting - you can enhance this based on your needs
        formatted = response.replace("**", "")
        formatted = formatted.replace("##", "\n🔹")
        formatted = formatted.replace("#", "\n📌")
        return formatted

    async def process_user_input(self, user_input: str) -> bool:
        """Process user input and return whether to continue the session."""
        user_input = user_input.strip()

        # Handle special commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye! Thanks for using the Risk Calculation Agent.")
            return False

        elif user_input.lower() in ["help", "h"]:
            self.print_help()
            return True

        elif user_input.lower() in ["example", "ex"]:
            self.print_example()
            return True

        elif not user_input:
            print("Please enter a question or command. Type 'help' for assistance.")
            return True

        # Process the question with the agent
        try:
            print("\n🤔 Thinking...")
            response = await get_response(user_input)

            # Extract and format the response
            response_content = extract_response_content(response)
            formatted_response = self.format_response(response_content)

            print(f"\n🤖 Agent Response:")
            print("-" * 50)
            print(formatted_response)
            print("-" * 50)

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'help' for assistance.")

        return True

    async def run(self):
        """Run the CLI main loop."""
        self.print_banner()

        while self.session_active:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()

                # Process the input
                continue_session = await self.process_user_input(user_input)

                if not continue_session:
                    self.session_active = False

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                print("Please try again.")


def main():
    """Main entry point for the CLI."""
    try:
        cli = RiskAgentCLI()
        asyncio.run(cli.run())
    except Exception as e:
        print(f"Failed to start CLI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
