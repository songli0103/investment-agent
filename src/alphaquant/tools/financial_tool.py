"""CrewAI tool for financial statements."""
from __future__ import annotations

from crewai.tools import BaseTool

from alphaquant.infrastructure.data_sources import DataSourceRegistry


TOOL_TIMEOUT_SECONDS = 30.0


class FinancialTool(BaseTool):
    name: str = "financial_statements_lookup"
    description: str = "Fetch financial statements (income statement, balance sheet, cash flow) for a US stock ticker."

    def _run(self, ticker: str) -> str:
        import asyncio

        registry = DataSourceRegistry()
        try:
            loop = asyncio.new_event_loop()
            try:
                statements = loop.run_until_complete(
                    asyncio.wait_for(
                        registry.get_financial(ticker),
                        timeout=TOOL_TIMEOUT_SECONDS,
                    )
                )
            finally:
                loop.close()
        except asyncio.TimeoutError:
            return f"Error fetching financials: timeout after {TOOL_TIMEOUT_SECONDS}s"
        except Exception as e:
            # Sub-3 Blocker 3: include exception type for diagnosability;
            # parse_crew_output's error-string detector still catches this prefix.
            return f"Error fetching financials: {type(e).__name__}: {e}"
        if not statements:
            return f"No financial data available for {ticker}"
        return statements.model_dump_json(indent=2)


__all__ = ["FinancialTool", "TOOL_TIMEOUT_SECONDS"]