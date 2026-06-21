"""CrewAI tool for company metadata lookup.

Wraps ``DataSourceRegistry.get_company`` and returns a JSON-serialized
``Company`` instance, or an error string if the call fails or times out.
Sub-project 2: this tool is what makes ``CompanyResolver`` a real data agent.
"""
from __future__ import annotations

import asyncio

from crewai.tools import BaseTool

from alphaquant.infrastructure.data_sources import DataSourceRegistry

# Per-tool fetch timeout. Whole-Flow timeout is set separately in
# ``flows/analysis_flow.py:FLOW_TIMEOUT_SECONDS``.
TOOL_TIMEOUT_SECONDS = 30.0


class CompanyLookupTool(BaseTool):
    name: str = "company_lookup"
    description: str = (
        "Resolve canonical company metadata (name, exchange, sector, industry, "
        "market cap) for a US stock ticker symbol."
    )

    def _run(self, ticker: str) -> str:
        registry = DataSourceRegistry()
        try:
            loop = asyncio.new_event_loop()
            try:
                company = loop.run_until_complete(
                    asyncio.wait_for(
                        registry.get_company(ticker),
                        timeout=TOOL_TIMEOUT_SECONDS,
                    )
                )
            finally:
                loop.close()
        except asyncio.TimeoutError:
            return f"Error fetching company: timeout after {TOOL_TIMEOUT_SECONDS}s"
        except Exception as e:
            # AllDataSourcesDown, network errors, validation errors, etc.
            return f"Error fetching company: {e}"
        if not company:
            return f"No company data available for {ticker}"
        return company.model_dump_json(indent=2)


__all__ = ["CompanyLookupTool"]