"""CrewAI DCF 估值假设工具。"""
from __future__ import annotations

from crewai.tools import BaseTool


class DCFTool(BaseTool):
    name: str = "dcf_assumptions"
    description: str = "提供 DCF 估值假设:增长率、WACC、终值增长率。返回默认值。"

    def _run(self, ticker: str) -> str:
        return (
            "DCF 假设(默认值 —— 分析师必须验证):\n"
            "- 增长率(5 年):8%\n"
            "- WACC:9%\n"
            "- 终值增长率:2.5%\n"
            "- 税率:21%\n"
            "这些是行业平均占位符。根据公司情况按需覆盖。"
        )


__all__ = ["DCFTool"]
