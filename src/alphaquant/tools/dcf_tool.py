"""CrewAI tool for DCF valuation assumptions."""
from __future__ import annotations

from crewai.tools import BaseTool


class DCFTool(BaseTool):
    name: str = "dcf_assumptions"
    description: str = "Provide DCF valuation assumptions: growth rate, WACC, terminal growth. Returns default values."

    def _run(self, ticker: str) -> str:
        return (
            "DCF assumptions (defaults — analyst must validate):\n"
            "- Growth rate (5yr): 8%\n"
            "- WACC: 9%\n"
            "- Terminal growth: 2.5%\n"
            "- Tax rate: 21%\n"
            "These are industry-average placeholders. Override per company as needed."
        )


__all__ = ["DCFTool"]
