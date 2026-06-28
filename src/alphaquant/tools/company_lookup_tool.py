"""CrewAI 公司元数据查询工具。

包装 ``DataSourceRegistry.get_company`` 并返回 JSON 序列化的 ``Company`` 实例,
如果调用失败或超时则返回错误字符串。
子项目 2:此工具使 ``CompanyResolver`` 成为真正的数据代理。
"""
from __future__ import annotations

import asyncio

from crewai.tools import BaseTool

from alphaquant.infrastructure.data_sources import DataSourceRegistry

# 每个工具的获取超时。整个 Flow 的超时单独设置在
# ``flows/analysis_flow.py:FLOW_TIMEOUT_SECONDS``。
TOOL_TIMEOUT_SECONDS = 30.0


class CompanyLookupTool(BaseTool):
    name: str = "company_lookup"
    description: str = (
        "解析美股股票代码的规范公司元数据(名称、交易所、行业分类、细分行业、市值)。"
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
            # Sub-3 Blocker 3: include exception type for diagnosability;
            # parse_crew_output's error-string detector still catches this prefix.
            return f"Error fetching company: {type(e).__name__}: {e}"
        if not company:
            return f"No company data available for {ticker}"
        return company.model_dump_json(indent=2)


__all__ = ["CompanyLookupTool"]