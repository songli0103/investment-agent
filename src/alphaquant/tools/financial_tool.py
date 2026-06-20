"""CrewAI tool for financial statements."""
from __future__ import annotations

from crewai.tools import BaseTool

from alphaquant.data_sources import DataSourceRegistry


class FinancialTool(BaseTool):
    name: str = "financial_statements_lookup"
    description: str = "Fetch financial statements (income statement, balance sheet, cash flow) for a US stock ticker."

    def _run(self, ticker: str) -> str:
        import asyncio

        registry = DataSourceRegistry()
        try:
            loop = asyncio.new_event_loop()
            statements = loop.run_until_complete(registry.get_financial(ticker))
            loop.close()
        except Exception as e:
            return f"Error fetching financials: {e}"
        if not statements:
            return f"No financial data available for {ticker}"
        return statements.model_dump_json(indent=2)


__all__ = ["FinancialTool"]
