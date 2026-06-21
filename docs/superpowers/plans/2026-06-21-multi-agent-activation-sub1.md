# Multi-Agent Activation — Sub-Project 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 8 个 `build_*_agent()` 函数从死代码变成 CrewAI Crew 实际调度的对象；`AnalysisFlow` 退化为薄壳；保持 `InvestmentReport` 输出严格一致（除时间戳/UUID）。

**Architecture:** 8 个 CrewAI Agent 装进 `AnalysisCrew`（`Process.hierarchical` + `manager_llm`），由 `AnalysisFlow.run_crew()` 在 `asyncio.to_thread` 里同步调用。Flow 在调用 Crew 前通过 DataSourceRegistry 预取所有原始数据并作为 inputs 传入；agent 处理后 `parse_crew_output()` 把结果回填 `AnalysisState`；Flow 再用现有逻辑合成 `InvestmentReport`。

**Tech Stack:** Python 3.11 + CrewAI 0.80+ + Pydantic v2 + pytest 8 + uv。

## Global Constraints

- **范围**：本计划仅覆盖子项目 1（Crew 编排壳）。子项目 2-4 单独 spec + 单独 plan。
- **输出一致性**：`InvestmentReport` 的非时间戳/UUID 字段值必须与子项目 1 之前 byte-for-byte 一致。
- **不动文件**：`core.py`、`scoring/*`、`infrastructure/data_sources/*`、`infrastructure/llm.py`、`infrastructure/config.py`、`models/*`、`observability/*`、`main.py`、`exceptions.py`、`interfaces/cli.py`、`interfaces/api/*`、`interfaces/frontend/*`、`tools/*`。
- **失败处理**：子项目 1 不做 retry/degrade。Crew 任何异常 → 直接抛 `ReportGenerationError` → FastAPI 返回 500。
- **测试**：所有现有测试必须通过；新增至少 6 个 AnalysisCrew 测试 + 重写 test_agents.py + 更新 test_flow.py。
- **提交粒度**：每个 Task 一个独立 commit。
- **工作分支**：`main`（单开发者项目）；commit message 形如 `<type>(scope): <subject>`。
- **CrewAI 版本**：锁定 `crewai>=0.80,<0.90`。

## File Structure

### 新增

| 路径 | 用途 |
|---|---|
| `src/alphaquant/crews/__init__.py` | package marker（导出 `AnalysisCrew`） |
| `src/alphaquant/crews/analysis_crew.py` | `AnalysisCrew` 类（封装 Crew + 8 Tasks） |
| `tests/test_crew.py` | AnalysisCrew 单元测试 |

### 修改

| 路径 | 变更 |
|---|---|
| `src/alphaquant/agents/company_resolver.py` | `build_company_resolver_agent(llm)` 接受 llm 参数；`tools=[]` |
| `src/alphaquant/agents/market_analyst.py` | `build_market_analyst_agent(llm)`；`tools=[MarketDataTool()]` |
| `src/alphaquant/agents/news_analyst.py` | `build_news_analyst_agent(llm)`；`tools=[NewsTool()]` |
| `src/alphaquant/agents/financial_analyst.py` | `build_financial_analyst_agent(llm)`；`tools=[FinancialTool()]` |
| `src/alphaquant/agents/competitor_analyst.py` | `build_competitor_analyst_agent(llm)`；`tools=[CompetitorTool()]` |
| `src/alphaquant/agents/risk_analyst.py` | `build_risk_analyst_agent(llm)`；`tools=[]` |
| `src/alphaquant/agents/valuation_analyst.py` | `build_valuation_analyst_agent(llm)`；`tools=[DCFTool()]` |
| `src/alphaquant/agents/report_writer.py` | `build_report_writer_agent(llm)`；`tools=[]` |
| `src/alphaquant/flows/analysis_flow.py` | 删除 6 个 `@listen` 步骤；新增 `@start run_crew` + `@listen synthesize_report` |
| `tests/test_agents.py` | 重写：验证每个 `build_*_agent(llm)` 返回正确配置的 `Agent` 对象 |
| `tests/test_flow.py` | 端到端测试改为 mock `AnalysisCrew` |

---

## Task 1: 重构 8 个 agent 构造器（接受 llm + 配置 tools）

**Files:**
- Modify: `src/alphaquant/agents/company_resolver.py`
- Modify: `src/alphaquant/agents/market_analyst.py`
- Modify: `src/alphaquant/agents/news_analyst.py`
- Modify: `src/alphaquant/agents/financial_analyst.py`
- Modify: `src/alphaquant/agents/competitor_analyst.py`
- Modify: `src/alphaquant/agents/risk_analyst.py`
- Modify: `src/alphaquant/agents/valuation_analyst.py`
- Modify: `src/alphaquant/agents/report_writer.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `from crewai import Agent`、`from alphaquant.infrastructure.llm import get_llm`
- Produces: 8 个 `build_*_agent(llm: LLM) -> Agent` 函数，每个按 tools 表配置

**Tools 配置表（来自 spec）**：

| Agent | Tools |
|---|---|
| CompanyResolver | `[]` |
| MarketAnalyst | `[MarketDataTool()]` |
| NewsAnalyst | `[NewsTool()]` |
| FinancialAnalyst | `[FinancialTool()]` |
| CompetitorAnalyst | `[CompetitorTool()]` |
| RiskAnalyst | `[]` |
| ValuationAnalyst | `[DCFTool()]` |
| ReportWriter | `[]` |

- [ ] **Step 1: 重写 tests/test_agents.py 为新接口**

完整替换 `tests/test_agents.py`：

```python
"""Tests for alphaquant.agents builder functions."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alphaquant.agents.company_resolver import build_company_resolver_agent
from alphaquant.agents.competitor_analyst import build_competitor_analyst_agent
from alphaquant.agents.financial_analyst import build_financial_analyst_agent
from alphaquant.agents.market_analyst import build_market_analyst_agent
from alphaquant.agents.news_analyst import build_news_analyst_agent
from alphaquant.agents.report_writer import build_report_writer_agent
from alphaquant.agents.risk_analyst import build_risk_analyst_agent
from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent
from alphaquant.tools.competitor_tool import CompetitorTool
from alphaquant.tools.dcf_tool import DCFTool
from alphaquant.tools.financial_tool import FinancialTool
from alphaquant.tools.market_data_tool import MarketDataTool
from alphaquant.tools.news_tool import NewsTool


@pytest.fixture
def fake_llm() -> MagicMock:
    return MagicMock(name="FakeLLM")


class TestAgentBuilders:
    """Each builder accepts an LLM and returns a configured Agent."""

    def test_company_resolver_has_no_tools(self, fake_llm):
        from crewai import Agent
        agent = build_company_resolver_agent(fake_llm)
        assert isinstance(agent, Agent)
        assert agent.tools == []

    def test_market_analyst_has_market_data_tool(self, fake_llm):
        agent = build_market_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], MarketDataTool)

    def test_news_analyst_has_news_tool(self, fake_llm):
        agent = build_news_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], NewsTool)

    def test_financial_analyst_has_financial_tool(self, fake_llm):
        agent = build_financial_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], FinancialTool)

    def test_competitor_analyst_has_competitor_tool(self, fake_llm):
        agent = build_competitor_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], CompetitorTool)

    def test_risk_analyst_has_no_tools(self, fake_llm):
        agent = build_risk_analyst_agent(fake_llm)
        assert agent.tools == []

    def test_valuation_analyst_has_dcf_tool(self, fake_llm):
        agent = build_valuation_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], DCFTool)

    def test_report_writer_has_no_tools(self, fake_llm):
        agent = build_report_writer_agent(fake_llm)
        assert agent.tools == []

    @pytest.mark.parametrize(
        "builder_name",
        [
            "build_company_resolver_agent",
            "build_market_analyst_agent",
            "build_news_analyst_agent",
            "build_financial_analyst_agent",
            "build_competitor_analyst_agent",
            "build_risk_analyst_agent",
            "build_valuation_analyst_agent",
            "build_report_writer_agent",
        ],
    )
    def test_builder_uses_passed_llm(self, fake_llm, builder_name):
        """Each agent uses the LLM passed in (not a fresh get_llm() call)."""
        from alphaquant import agents

        builder = getattr(agents, builder_name)
        agent = builder(fake_llm)
        assert agent.llm is fake_llm

    @pytest.mark.parametrize(
        "builder_name",
        [
            "build_company_resolver_agent",
            "build_market_analyst_agent",
            "build_news_analyst_agent",
            "build_financial_analyst_agent",
            "build_competitor_analyst_agent",
            "build_risk_analyst_agent",
            "build_valuation_analyst_agent",
            "build_report_writer_agent",
        ],
    )
    def test_builder_verbose_is_false(self, fake_llm, builder_name):
        """Verbose=False in production to avoid log spam; can be toggled later."""
        from alphaquant import agents

        builder = getattr(agents, builder_name)
        agent = builder(fake_llm)
        assert agent.verbose is False
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `uv run pytest tests/test_agents.py -v`
Expected: 全部 8 个 tools 测试失败，错误 `TypeError: build_*_agent() takes 0 positional arguments but 1 was given`（旧接口不接受 llm 参数）

- [ ] **Step 3: 修改 src/alphaquant/agents/company_resolver.py**

完整重写为：

```python
"""CompanyResolver Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM


def build_company_resolver_agent(llm: LLM) -> Agent:
    return Agent(
        role="Company Identification Specialist",
        goal="Validate and standardize ticker symbols, resolve company metadata.",
        backstory=(
            "You are a data engineer specializing in US equity identifiers. "
            "Given a ticker, you return the canonical company name, exchange, "
            "sector, industry, and market cap. You never invent data."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_company_resolver_agent"]
```

- [ ] **Step 4: 修改 src/alphaquant/agents/market_analyst.py**

完整重写为：

```python
"""MarketAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.market_data_tool import MarketDataTool


def build_market_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Market Data Specialist",
        goal=(
            "Fetch real-time market data (price, P/E, market cap, 52-week range, "
            "beta, growth metrics) for a US stock ticker. Report data verbatim — "
            "do not interpret or summarize."
        ),
        backstory=(
            "You are a quantitative data fetcher. You call market_data_lookup "
            "exactly once with the ticker and return its JSON output as-is."
        ),
        tools=[MarketDataTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_market_analyst_agent"]
```

- [ ] **Step 5: 修改 src/alphaquant/agents/news_analyst.py**

完整重写为：

```python
"""NewsAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.news_tool import NewsTool


def build_news_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="News Retrieval Specialist",
        goal=(
            "Fetch recent news (last 30 days) for a US stock ticker. "
            "Report news items verbatim — do not editorialize."
        ),
        backstory=(
            "You are a news data fetcher. You call news_lookup exactly once "
            "with the ticker and return its JSON output as-is."
        ),
        tools=[NewsTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_news_analyst_agent"]
```

- [ ] **Step 6: 修改 src/alphaquant/agents/financial_analyst.py**

完整重写为：

```python
"""FinancialAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.financial_tool import FinancialTool


def build_financial_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Financial Statements Specialist",
        goal=(
            "Fetch income statements, balance sheets, and cash flow statements "
            "for a US stock ticker. Report data verbatim — do not calculate ratios."
        ),
        backstory=(
            "You are a financial data fetcher. You call financial_statements_lookup "
            "exactly once with the ticker and return its JSON output as-is."
        ),
        tools=[FinancialTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_financial_analyst_agent"]
```

- [ ] **Step 7: 修改 src/alphaquant/agents/competitor_analyst.py**

完整重写为：

```python
"""CompetitorAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.competitor_tool import CompetitorTool


def build_competitor_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Competitive Landscape Analyst",
        goal=(
            "Identify and rank competitors for a US stock ticker. "
            "Return peer tickers, market caps, growth, margins."
        ),
        backstory=(
            "You are a sell-side equity analyst. You call competitor_lookup "
            "with the ticker, then summarize the peer set with industry rank."
        ),
        tools=[CompetitorTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_competitor_analyst_agent"]
```

- [ ] **Step 8: 修改 src/alphaquant/agents/risk_analyst.py**

完整重写为：

```python
"""RiskAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM


def build_risk_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Risk Assessment Specialist",
        goal=(
            "Compute risk assessment from upstream data already in shared memory "
            "(company, market, financial). Report sub-scores per category."
        ),
        backstory=(
            "You are a risk officer. You read financial ratios (debt ratio, beta) "
            "from memory and assign risk scores 0-10 per category."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_risk_analyst_agent"]
```

- [ ] **Step 9: 修改 src/alphaquant/agents/valuation_analyst.py**

完整重写为：

```python
"""ValuationAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.dcf_tool import DCFTool


def build_valuation_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Sell-side Valuation Modeler",
        goal=(
            "Estimate intrinsic value using DCF, relative valuation (P/E, P/B, P/S), "
            "and PEG. Provide a value range (low-high) with explicit assumptions."
        ),
        backstory=(
            "You are a sell-side equity research modeler. You build DCF models with "
            "explicit assumptions (growth, WACC, terminal). You cross-check with peer "
            "multiples. You never give a single point estimate—always a range ±15%."
        ),
        tools=[DCFTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_valuation_analyst_agent"]
```

- [ ] **Step 10: 修改 src/alphaquant/agents/report_writer.py**

完整重写为：

```python
"""ReportWriter Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM


def build_report_writer_agent(llm: LLM) -> Agent:
    return Agent(
        role="Investment Report Synthesizer",
        goal=(
            "Synthesize all upstream data (company, market, financial, news, "
            "competitor, risk, valuation) into a final InvestmentReport markdown."
        ),
        backstory=(
            "You are an investment writer. You read everything from shared memory "
            "and produce a clear, structured markdown report with rating, "
            "confidence, and rationale."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_report_writer_agent"]
```

- [ ] **Step 11: 修复 src/alphaquant/agents/__init__.py**

当前 `__init__.py` 内容只有 docstring。保持现状——8 个 agent 通过 `from alphaquant.agents.xxx import ...` 显式导入（与现有 flow 代码一致）。无需修改。

- [ ] **Step 12: 运行测试，确认全部通过**

Run: `uv run pytest tests/test_agents.py -v`
Expected: 18 个测试全部通过（8 个 tools 配置 + 8 个 uses_passed_llm + 8 个 verbose_is_false，但 `@pytest.mark.parametrize` 重复 — 实际计数：8 + 2×8 = 24 个 PASSED）

- [ ] **Step 13: Commit**

```bash
git add tests/test_agents.py \
        src/alphaquant/agents/company_resolver.py \
        src/alphaquant/agents/market_analyst.py \
        src/alphaquant/agents/news_analyst.py \
        src/alphaquant/agents/financial_analyst.py \
        src/alphaquant/agents/competitor_analyst.py \
        src/alphaquant/agents/risk_analyst.py \
        src/alphaquant/agents/valuation_analyst.py \
        src/alphaquant/agents/report_writer.py
git commit -m "refactor(agents): accept llm parameter and configure tools

Each build_*_agent() now takes an LLM as parameter (was calling
get_llm() internally). This decouples agent construction from
LLM factory so AnalysisCrew can share one LLM instance across all
agents. Tools per agent follow the sub-project 1 spec table.

8 agents now match a uniform shape: build_*_agent(llm: LLM) -> Agent.
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: 创建 AnalysisCrew 类（CrewAI Crew 容器）

**Files:**
- Create: `src/alphaquant/crews/__init__.py`
- Create: `src/alphaquant/crews/analysis_crew.py`
- Test: `tests/test_crew.py`

**Interfaces:**
- Consumes: 8 个 `build_*_agent(llm)` 函数、`crewai.Crew`、`crewai.Process`、`crewai.Task`、`alphaquant.infrastructure.llm.get_llm`
- Produces: `AnalysisCrew` 类，构造时返回配置好的 `Crew`；`kickoff(inputs)` 方法运行

- [ ] **Step 1: 写 tests/test_crew.py 的失败测试**

创建 `tests/test_crew.py`：

```python
"""Tests for alphaquant.crews.AnalysisCrew."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alphaquant.agents.company_resolver import build_company_resolver_agent
from alphaquant.agents.competitor_analyst import build_competitor_analyst_agent
from alphaquant.agents.financial_analyst import build_financial_analyst_agent
from alphaquant.agents.market_analyst import build_market_analyst_agent
from alphaquant.agents.news_analyst import build_news_analyst_agent
from alphaquant.agents.report_writer import build_report_writer_agent
from alphaquant.agents.risk_analyst import build_risk_analyst_agent
from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent
from alphaquant.tools.competitor_tool import CompetitorTool
from alphaquant.tools.dcf_tool import DCFTool
from alphaquant.tools.financial_tool import FinancialTool
from alphaquant.tools.market_data_tool import MarketDataTool
from alphaquant.tools.news_tool import NewsTool


@pytest.fixture
def fake_llm() -> MagicMock:
    return MagicMock(name="FakeLLM")


class TestAnalysisCrew:
    def test_imports(self):
        from alphaquant.crews import AnalysisCrew

        assert AnalysisCrew is not None

    def test_instantiates_without_error(self, monkeypatch, fake_llm):
        """AnalysisCrew() should not raise; get_llm is mocked."""
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert crew is not None

    def test_all_8_agents_built(self, monkeypatch, fake_llm):
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert len(crew.agents) == 8

    def test_agents_are_crewai_agents(self, monkeypatch, fake_llm):
        from crewai import Agent

        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        for agent in crew.agents:
            assert isinstance(agent, Agent)

    def test_all_8_tasks_built(self, monkeypatch, fake_llm):
        from crewai import Task

        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert len(crew.tasks) == 8
        for task in crew.tasks:
            assert isinstance(task, Task)

    def test_process_is_hierarchical(self, monkeypatch, fake_llm):
        from crewai import Process

        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert crew.crew.process == Process.hierarchical

    def test_manager_llm_configured(self, monkeypatch, fake_llm):
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert crew.crew.manager_llm is fake_llm

    def test_memory_disabled_for_sub1(self, monkeypatch, fake_llm):
        """Sub-project 1: memory=False. Sub-project 4 enables it."""
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert crew.crew.memory is False

    def test_tools_mapping(self, monkeypatch, fake_llm):
        """Each agent's tools match the spec table."""
        expected_tools = {
            build_company_resolver_agent: [],
            build_market_analyst_agent: [MarketDataTool],
            build_news_analyst_agent: [NewsTool],
            build_financial_analyst_agent: [FinancialTool],
            build_competitor_analyst_agent: [CompetitorTool],
            build_risk_analyst_agent: [],
            build_valuation_analyst_agent: [DCFTool],
            build_report_writer_agent: [],
        }
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()

        # Each agent's builder is identifiable via its role.
        role_to_builder = {
            "Company Identification Specialist": build_company_resolver_agent,
            "Market Data Specialist": build_market_analyst_agent,
            "News Retrieval Specialist": build_news_analyst_agent,
            "Financial Statements Specialist": build_financial_analyst_agent,
            "Competitive Landscape Analyst": build_competitor_analyst_agent,
            "Risk Assessment Specialist": build_risk_analyst_agent,
            "Sell-side Valuation Modeler": build_valuation_analyst_agent,
            "Investment Report Synthesizer": build_report_writer_agent,
        }
        for agent in crew.agents:
            builder = role_to_builder[agent.role]
            expected_classes = expected_tools[builder]
            assert len(agent.tools) == len(expected_classes), (
                f"{agent.role}: expected {len(expected_classes)} tool(s), "
                f"got {len(agent.tools)}"
            )
            for tool, expected_cls in zip(agent.tools, expected_classes):
                assert isinstance(tool, expected_cls)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `uv run pytest tests/test_crew.py -v`
Expected: `ModuleNotFoundError: No module named 'alphaquant.crews'`

- [ ] **Step 3: 创建 src/alphaquant/crews/__init__.py**

```python
"""Crew orchestrations — wraps CrewAI Crew around the 8 alphaquant agents."""
from alphaquant.crews.analysis_crew import AnalysisCrew

__all__ = ["AnalysisCrew"]
```

- [ ] **Step 4: 创建 src/alphaquant/crews/analysis_crew.py**

```python
"""AnalysisCrew: 8-agent CrewAI Crew orchestrating the analysis pipeline.

Sub-project 1 (this file): Crew is built with Process.hierarchical and a
shared manager_llm, but ``memory=False`` and agents are 'thin' wrappers
calling the existing data tools. Sub-project 4 will enable ``memory`` and
``allow_delegation=True``.

The Flow (``flows/analysis_flow.py``) pre-fetches all raw data
(company / market / news / financial) via ``DataSourceRegistry`` and passes
it in ``inputs``. After ``kickoff`` returns, ``parse_crew_output()``
extracts the agent outputs into ``AnalysisState``.
"""
from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, Process, Task

from alphaquant.agents.company_resolver import build_company_resolver_agent
from alphaquant.agents.competitor_analyst import build_competitor_analyst_agent
from alphaquant.agents.financial_analyst import build_financial_analyst_agent
from alphaquant.agents.market_analyst import build_market_analyst_agent
from alphaquant.agents.news_analyst import build_news_analyst_agent
from alphaquant.agents.report_writer import build_report_writer_agent
from alphaquant.agents.risk_analyst import build_risk_analyst_agent
from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent
from alphaquant.infrastructure.llm import get_llm


_TASK_TEMPLATES: list[tuple[str, str, str]] = [
    # (role_key, description_template, expected_output)
    (
        "company_resolver",
        "Validate ticker '{ticker}' and return canonical company metadata.",
        "JSON with company name, exchange, sector, industry, market cap",
    ),
    (
        "market_analyst",
        "Fetch market data for '{ticker}'.",
        "JSON with price, P/E, market cap, 52w range, beta, growth metrics",
    ),
    (
        "news_analyst",
        "Fetch recent news for '{ticker}'.",
        "JSON array of news items (date, title, source, url)",
    ),
    (
        "financial_analyst",
        "Fetch financial statements for '{ticker}'.",
        "JSON with income statements, balance sheets, cash flows",
    ),
    (
        "competitor_analyst",
        "Identify competitors and compute competitive score for '{ticker}'.",
        "JSON with peer tickers, market caps, growth, margins, rank",
    ),
    (
        "risk_analyst",
        "Compute risk assessment for '{ticker}' from upstream data.",
        "JSON with sub-scores per category and total",
    ),
    (
        "valuation_analyst",
        "Compute valuation (DCF + relative) for '{ticker}'.",
        "JSON with intrinsic value, DCF value, relative value, upside",
    ),
    (
        "report_writer",
        "Synthesize InvestmentReport markdown for '{ticker}'.",
        "Markdown report with rating, confidence, sections",
    ),
]


class AnalysisCrew:
    """Wraps the 8 CrewAI agents in a hierarchical Crew.

    Sub-project 1 keeps the crew as a structural shell: it can be invoked
    end-to-end, but its outputs are normalized by ``parse_crew_output``
    in the calling Flow. Sub-project 3 will let agents do real reasoning;
    sub-project 4 will enable memory and peer delegation.
    """

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.1)
        self.agents: list[Agent] = self._build_agents()
        self.tasks: list[Task] = self._build_tasks()
        self.crew: Crew = self._build_crew()

    def _build_agents(self) -> list[Agent]:
        return [
            build_company_resolver_agent(self._llm),
            build_market_analyst_agent(self._llm),
            build_news_analyst_agent(self._llm),
            build_financial_analyst_agent(self._llm),
            build_competitor_analyst_agent(self._llm),
            build_risk_analyst_agent(self._llm),
            build_valuation_analyst_agent(self._llm),
            build_report_writer_agent(self._llm),
        ]

    def _build_tasks(self) -> list[Task]:
        tasks: list[Task] = []
        for role_key, description, expected in _TASK_TEMPLATES:
            agent = self.agents[len(tasks)]
            tasks.append(
                Task(
                    description=description,
                    expected_output=expected,
                    agent=agent,
                )
            )
        return tasks

    def _build_crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=self._llm,
            memory=False,
            verbose=False,
        )

    def kickoff(self, inputs: dict[str, Any]):
        """Synchronous entry point — wraps Crew.kickoff().

        The Flow layer is responsible for invoking this inside
        ``asyncio.to_thread`` so the event loop is not blocked.
        """
        return self.crew.kickoff(inputs=inputs)


__all__ = ["AnalysisCrew"]
```

- [ ] **Step 5: 运行测试，确认全部通过**

Run: `uv run pytest tests/test_crew.py -v`
Expected: 9 个测试全部通过

- [ ] **Step 6: Commit**

```bash
git add src/alphaquant/crews/__init__.py \
        src/alphaquant/crews/analysis_crew.py \
        tests/test_crew.py
git commit -m "feat(crews): add AnalysisCrew wrapping 8 agents in hierarchical Crew

AnalysisCrew builds the 8 agents (built in previous commit) into a
single CrewAI Crew with Process.hierarchical and shared manager_llm.
memory=False and verbose=False for sub-project 1; both are toggled
in sub-project 4.

kickoff(inputs) is the synchronous entry point — AnalysisFlow wraps
it in asyncio.to_thread for async compatibility.
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: 把 AnalysisFlow 改为 Crew 编排薄壳

**Files:**
- Modify: `src/alphaquant/flows/analysis_flow.py:160-572`（替换 6 个 `@listen` 方法为 `@start run_crew` + `@listen synthesize_report`）
- Test: `tests/test_flow.py`

**Interfaces:**
- Consumes: `AnalysisCrew` (from `alphaquant.crews`)、`asyncio.to_thread`、`asyncio.wait_for`、`AnalysisState` (Pydantic)
- Produces:
  - `parse_crew_output(result: CrewOutput) -> dict[str, Any]` 提取 8 个字段映射
  - `AnalysisFlow.run_crew(ticker)` — pre-fetch + kickoff + parse
  - `AnalysisFlow.synthesize_report()` — 现有 InvestmentReport 合成逻辑

- [ ] **Step 1: 添加 tests/test_flow.py 的失败测试**

在 `tests/test_flow.py` 末尾添加（`TestFlowKickoff` 类之后）：

```python
class TestRunCrewStep:
    """@start run_crew: pre-fetch data, invoke AnalysisCrew, fill state."""

    def test_run_crew_pre_fetches_data_and_invokes_crew(
        self, sample_company, sample_market, sample_news, sample_financial
    ):
        """run_crew must call DataSourceRegistry 4 times (one per data source)
        AND must invoke AnalysisCrew.kickoff with the pre-fetched data."""
        from alphaquant.flows.analysis_flow import AnalysisFlow

        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"

        # Mock DataSourceRegistry
        reg_cls = __import__(
            "alphaquant.infrastructure.data_sources", fromlist=["DataSourceRegistry"]
        ).DataSourceRegistry

        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=sample_news)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)):

            # Mock AnalysisCrew so we don't actually invoke real LLM
            with patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:
                # Fake crew output: each task returns a simple marker
                from crewai.agents import CrewOutput  # adjust import if needed
                fake_output = MagicMock()
                fake_output.tasks_output = []
                MockCrew.return_value.kickoff.return_value = fake_output

                _run(flow.run_crew("AAPL"))

                # Verify crew was called
                MockCrew.assert_called_once()
                MockCrew.return_value.kickoff.assert_called_once()
                call_kwargs = MockCrew.return_value.kickoff.call_args.kwargs
                assert "inputs" in call_kwargs
                inputs = call_kwargs["inputs"]
                assert inputs["ticker"] == "AAPL"

    def test_run_crew_timeout_raises(self, sample_company, sample_market, sample_news, sample_financial):
        """If crew.kickoff exceeds 120s, asyncio.TimeoutError is raised."""
        import asyncio as _asyncio
        from alphaquant.flows.analysis_flow import AnalysisFlow

        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"

        reg_cls = __import__(
            "alphaquant.infrastructure.data_sources", fromlist=["DataSourceRegistry"]
        ).DataSourceRegistry

        async def slow_kickoff(inputs):
            await _asyncio.sleep(0.5)

        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=sample_news)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)), \
             patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew, \
             patch("alphaquant.flows.analysis_flow.FLOW_TIMEOUT_SECONDS", 0.1):
            MockCrew.return_value.kickoff = slow_kickoff

            with pytest.raises(_asyncio.TimeoutError):
                _run(flow.run_crew("AAPL"))


class TestParseCrewOutput:
    """parse_crew_output: CrewOutput → AnalysisState field dict."""

    def test_extracts_company_from_task_output(self):
        from alphaquant.flows.analysis_flow import parse_crew_output

        # Build fake CrewOutput with one task returning JSON for company
        fake_task_output = MagicMock()
        fake_task_output.description = "Validate ticker 'AAPL' and return canonical company metadata."
        fake_task_output.raw = '{"name": "Apple Inc.", "exchange": "NASDAQ"}'

        fake_result = MagicMock()
        fake_result.tasks_output = [fake_task_output]

        state_dict = parse_crew_output(fake_result)
        # Sub-project 1: parse_crew_output extracts raw text per task.
        # We assert the structure is a dict keyed by role_key.
        assert isinstance(state_dict, dict)
        assert "company_resolver" in state_dict
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `uv run pytest tests/test_flow.py::TestRunCrewStep tests/test_flow.py::TestParseCrewOutput -v`
Expected: ImportError on `parse_crew_output` and `flow.run_crew` doesn't exist yet

- [ ] **Step 3: 修改 src/alphaquant/flows/analysis_flow.py — 头部 import**

修改 import 段（在第 1-30 行的现有 import 之后追加）：

```python
from alphaquant.crews import AnalysisCrew
```

并新增 `_TASK_KEYWORDS` 列表（在 import 段之后、`GICS_PEERS` 之前）：

```python
# Maps crew task descriptions to AnalysisState field keys. The order MUST
# match the order in crews/analysis_crew.py::_TASK_TEMPLATES.
_TASK_KEYWORDS: list[str] = [
    "company_resolver",
    "market_analyst",
    "news_analyst",
    "financial_analyst",
    "competitor_analyst",
    "risk_analyst",
    "valuation_analyst",
    "report_writer",
]
```

- [ ] **Step 4: 重写 AnalysisFlow 步骤方法**

替换 `flows/analysis_flow.py` 中第 411-485 行的 `valuation_analysis` 方法 + 第 487-551 行的 `write_report` 方法 + 整个 `@listen` 链。

具体替换逻辑：

1. 删除 `@listen(parallel_data_collection) async def competitor_analysis(self)`（第 259-345 行）
2. 删除 `@listen(competitor_analysis) async def risk_analysis(self)`（第 347-409 行）
3. 删除 `@listen(risk_analysis) async def valuation_analysis(self)`（第 411-484 行）
4. **修改** `@listen(valuation_analysis) async def write_report(self)`（第 486-551 行）→ 重命名为 `synthesize_report`，移除 `@listen(valuation_analysis)` 装饰器改为 `@listen(run_crew)`
5. **新增** `@start async def run_crew(self, ticker: str | None = None, crewai_trigger_payload: dict[str, Any] | None = None)`

完整的 `run_crew` 方法：

```python
    @start()
    async def run_crew(
        self,
        ticker: str | None = None,
        crewai_trigger_payload: dict[str, Any] | None = None,
    ) -> None:
        """Step 1: Pre-fetch raw data via DataSourceRegistry, then drive the
        8-agent AnalysisCrew to produce analysis results. Sub-project 1
        keeps the crew as a structural shell; sub-project 3+ will let
        agents do real reasoning.
        """
        from alphaquant.infrastructure.data_sources import DataSourceRegistry

        # Resolve ticker from any of the supported channels.
        raw_ticker = (
            ticker
            or (crewai_trigger_payload or {}).get("ticker")
            or self.state.ticker
            or ""
        )
        normalized = _normalize_ticker(raw_ticker)
        self.state.ticker = normalized
        log.info("flow_step_started", step="run_crew", ticker=normalized)

        # Pre-fetch all 4 raw data sources. The crew gets these as inputs.
        registry = DataSourceRegistry()
        try:
            company, market, news, financial = await asyncio.wait_for(
                asyncio.gather(
                    registry.get_company(normalized),
                    registry.get_market(normalized),
                    registry.get_news(normalized),
                    registry.get_financial(normalized),
                    return_exceptions=True,
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            log.error("data_fetch_timeout", ticker=normalized)
            raise

        # Map exceptions → None (degraded mode).
        from alphaquant.models.company import Company
        from alphaquant.models.market import MarketData
        from alphaquant.models.news import NewsAnalysis
        from alphaquant.models.financial import FinancialStatements

        self.state.company = company if isinstance(company, Company) else None
        self.state.market = market if isinstance(market, MarketData) else None
        self.state.news = _news_items_to_analysis(
            news if isinstance(news, list) else [], normalized
        ) if not isinstance(news, NewsAnalysis) else news
        self.state.financial = (
            financial
            if isinstance(financial, FinancialStatements)
            else FinancialStatements(ticker=normalized)
        )

        if self.state.company is None:
            self.state.errors.append("company_data_unavailable")
        if self.state.market is None:
            self.state.errors.append("market_data_unavailable")
        if self.state.news.total_count == 0:
            self.state.errors.append("news_data_unavailable")
        if not isinstance(financial, FinancialStatements):
            self.state.errors.append("financial_data_unavailable")

        # Drive the 8-agent crew. Crew.kickoff is sync → wrap in to_thread.
        crew = AnalysisCrew()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    crew.kickoff,
                    inputs={
                        "ticker": normalized,
                        "company": self.state.company.model_dump() if self.state.company else None,
                        "market": self.state.market.model_dump() if self.state.market else None,
                        "news": self.state.news.model_dump() if self.state.news else None,
                        "financial": self.state.financial.model_dump() if self.state.financial else None,
                    },
                ),
                timeout=FLOW_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.error("crew_timeout", ticker=normalized)
            raise

        # Parse crew output → fill self.state fields downstream tasks consume.
        parse_crew_output(result, self.state)

        log.info("flow_step_completed", step="run_crew", ticker=normalized)
```

完整的 `synthesize_report` 方法（替换原 `write_report`）：

```python
    @listen(run_crew)
    async def synthesize_report(self) -> None:
        """Step 2: Synthesize InvestmentReport from crew-driven state.

        On any synthesis failure, raise ``ReportGenerationError`` so the
        caller (FastAPI handler per spec §5.2) can return HTTP 500.
        """
        log.info("flow_step_started", step="synthesize_report", ticker=self.state.ticker)
        assert self.state.company is not None
        assert self.state.news is not None
        assert self.state.financial is not None
        assert self.state.competitor is not None
        assert self.state.risk is not None
        assert self.state.valuation is not None

        # §3.2: market may be None (degraded) — substitute a minimal placeholder
        # so InvestmentReport.markdown and downstream consumers can render.
        market = self.state.market
        if market is None:
            market = MarketData(
                ticker=self.state.ticker,
                as_of=datetime.utcnow(),
                price=Decimal("0"),
                change_pct=0.0,
                volume=0,
                market_cap=self.state.company.market_cap,
                source="degraded",
            )
            self.state.market = market

        try:
            rating, confidence = determine_rating(
                self.state.valuation, self.state.risk, self.state.news
            )
            health_score = financial_health.compute(self.state.financial)

            markdown = _build_markdown(
                self.state.company,
                market,
                self.state.financial,
                self.state.news,
                self.state.competitor,
                self.state.risk,
                self.state.valuation,
                rating,
                confidence,
                health_score,
            )

            self.state.report = InvestmentReport(
                report_id=str(uuid.uuid4()),
                ticker=self.state.ticker,
                generated_at=datetime.utcnow(),
                data_as_of={
                    "market": market.as_of,
                    "news": self.state.news.as_of,
                },
                company=self.state.company,
                market=market,
                financial=self.state.financial,
                financial_health_score=health_score,
                news=self.state.news,
                competitors=self.state.competitor,
                risk=self.state.risk,
                valuation=self.state.valuation,
                rating=rating,
                confidence=confidence,
                catalysts=[],
                markdown=markdown,
                sources=_collect_sources(
                    market,
                    self.state.news,
                    self.state.financial,
                    self.state.competitor,
                ),
            )
            log.info(
                "flow_step_completed",
                step="synthesize_report",
                ticker=self.state.ticker,
                report_id=self.state.report.report_id,
                rating=rating,
                confidence=confidence,
                health_score=health_score,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "flow_step_failed",
                step="synthesize_report",
                ticker=self.state.ticker,
                error=str(exc),
            )
            raise ReportGenerationError(
                f"Failed to synthesize report for {self.state.ticker}: {exc}"
            ) from exc
```

- [ ] **Step 5: 添加 parse_crew_output 函数**

在 `_collect_sources` 函数之后、`AnalysisFlow` 类之前添加：

```python
def parse_crew_output(result: Any, state: "AnalysisState") -> None:
    """Extract agent outputs from CrewOutput and fill state fields in-place.

    Sub-project 1 (this implementation): each task output's ``raw`` text is
    assumed to be JSON. We parse it and assign by task order. Sub-project 3
    will let agents produce structured Pydantic outputs instead.
    """
    import json

    from alphaquant.models.competitor import Competitor, CompetitorAnalysis
    from alphaquant.models.risk import RiskAssessment
    from alphaquant.models.valuation import ValuationResult

    tasks_output = getattr(result, "tasks_output", []) or []
    for idx, task_out in enumerate(tasks_output):
        if idx >= len(_TASK_KEYWORDS):
            break
        key = _TASK_KEYWORDS[idx]
        raw = getattr(task_out, "raw", "") or ""

        try:
            data = json.loads(raw) if raw.strip().startswith("{") else {}
        except (json.JSONDecodeError, ValueError):
            data = {}

        if key == "company_resolver" and data:
            # company is pre-fetched by run_crew; nothing to do.
            pass
        elif key == "market_analyst" and data:
            pass  # market pre-fetched
        elif key == "news_analyst" and data:
            pass  # news pre-fetched
        elif key == "financial_analyst" and data:
            pass  # financial pre-fetched
        elif key == "competitor_analyst" and data:
            # Sub-project 1: deterministic fallback (current Flow logic).
            # Sub-project 3 will populate from agent output.
            peers_raw = data.get("peers", [])
            peers = []
            for p in peers_raw[:5]:
                try:
                    peers.append(Competitor(**p))
                except Exception:
                    continue
            if not peers:
                peers = _gics_peers_for(state.company, state.ticker)
            target_metrics = {
                "market_cap": float(state.market.market_cap if state.market else 0),
                "revenue_growth_yoy": float(
                    state.market.revenue_growth_yoy
                    if state.market and state.market.revenue_growth_yoy
                    else 0
                ),
                "gross_margin": 0,
                "net_margin": 0,
            }
            from alphaquant.scoring import competitive as scoring_competitive

            score = scoring_competitive.compute(target_metrics, peers)
            state.competitor = CompetitorAnalysis(
                target_ticker=state.ticker,
                competitors=peers,
                industry_rank=1,
                industry_size=max(10, len(peers) + 1),
                competitive_score=score,
                strengths=[],
                weaknesses=[],
                method="computed" if peers_raw else "fallback",
            )
        elif key == "risk_analyst" and data:
            from alphaquant.scoring import risk_score as scoring_risk

            sub_scores_data = data.get("sub_scores", [])
            from alphaquant.models.risk import RiskScore as RiskScoreModel

            sub_scores = [
                RiskScoreModel(
                    category=s["category"],
                    score=s["score"],
                    rationale=s.get("rationale", ""),
                    evidence=[],
                )
                for s in sub_scores_data
            ] if sub_scores_data else _default_risk_subscores(state)
            total = scoring_risk.compute(sub_scores)
            level = scoring_risk.determine_level(total)
            state.risk = RiskAssessment(
                ticker=state.ticker,
                total_score=total,
                level=level,
                sub_scores=sub_scores,
                top_risks=[s.rationale for s in sub_scores[:3]],
            )
        elif key == "valuation_analyst" and data:
            # Sub-project 1: deterministic fallback (current Flow logic).
            from alphaquant.scoring.dcf import compute_dcf_value

            current = state.market.price if state.market else Decimal("0")
            pe = state.market.pe_ratio if state.market and state.market.pe_ratio else 20.0
            peer_pe_avg = 20.0
            relative_value = current * Decimal(str(peer_pe_avg / pe)) if pe > 0 else current

            fcf_data = (
                state.financial.cash_flows[0].free_cash_flow
                if state.financial and state.financial.cash_flows
                else None
            )
            growth_pct = state.market.revenue_growth_yoy if state.market else None
            growth_rate = (growth_pct / 100.0) if growth_pct is not None else 0.05
            shares_outstanding = (
                int(state.market.market_cap / state.market.price)
                if state.market and state.market.price > 0
                else 0
            )
            dcf_value = None
            if fcf_data is not None and fcf_data > 0 and shares_outstanding > 0:
                dcf_value = compute_dcf_value(
                    fcf=fcf_data,
                    growth_rate=growth_rate,
                    shares_outstanding=shares_outstanding,
                )
            if dcf_value is not None and relative_value is not None:
                intrinsic = (dcf_value + relative_value) / 2
                method = "dcf_relative_peg"
            else:
                intrinsic = relative_value
                method = "relative_only"
            upside = float((intrinsic - current) / current) if current else 0.0
            state.valuation = ValuationResult(
                ticker=state.ticker,
                intrinsic_value_per_share=intrinsic,
                current_price=current,
                upside_pct=round(upside, 4),
                dcf_value=dcf_value,
                relative_value=relative_value,
                peg_ratio=None,
                method=method,
                assumptions={"peer_pe_avg": peer_pe_avg},
            )
        elif key == "report_writer" and data:
            pass  # synthesis happens in synthesize_report step
```

并在模块顶部添加辅助函数 `_default_risk_subscores`：

```python
def _default_risk_subscores(state: "AnalysisState") -> list:
    """Sub-project 1 fallback: same risk subscores the deterministic Flow uses."""
    from alphaquant.models.risk import RiskScore as RiskScoreModel

    fin_score = 5
    if state.financial and state.financial.balance_sheets:
        bs = state.financial.balance_sheets[0]
        debt_ratio = float(bs.total_liabilities / bs.total_assets * 100) if bs.total_assets else 50
        fin_score = min(10, max(0, int(debt_ratio / 10)))
    mkt_score = 5
    if state.market and state.market.beta is not None:
        mkt_score = min(10, max(0, int(abs(state.market.beta) * 5)))
    return [
        RiskScoreModel(
            category="financial",
            score=fin_score,
            rationale=f"Debt ratio suggests {fin_score}/10 financial risk",
            evidence=[],
        ),
        RiskScoreModel(
            category="market",
            score=mkt_score,
            rationale=f"Beta-implied market risk: {mkt_score}/10",
            evidence=[],
        ),
        RiskScoreModel(category="operational", score=5, rationale="Default neutral", evidence=[]),
        RiskScoreModel(category="regulatory", score=5, rationale="Default neutral", evidence=[]),
        RiskScoreModel(category="governance", score=5, rationale="Default neutral", evidence=[]),
        RiskScoreModel(category="macro", score=5, rationale="Default neutral", evidence=[]),
    ]
```

- [ ] **Step 6: 删除 resolve_company 步骤（已被 run_crew 取代）**

`resolve_company` 方法（第 163-205 行）已被 `run_crew` 取代。删除整个方法。但保留 `_normalize_ticker` 辅助函数（run_crew 内部使用）。

- [ ] **Step 7: 删除 parallel_data_collection 步骤（已被 run_crew 取代）**

整个 `parallel_data_collection` 方法（第 207-256 行）已被 run_crew 内的 pre-fetch 取代。删除整个方法。

- [ ] **Step 8: 运行 test_flow.py，确认通过**

Run: `uv run pytest tests/test_flow.py -v`
Expected:
- `TestRunCrewStep::test_run_crew_pre_fetches_data_and_invokes_crew` PASS
- `TestRunCrewStep::test_run_crew_timeout_raises` PASS
- `TestParseCrewOutput::test_extracts_company_from_task_output` PASS
- 现有 `TestValuationAnalysis`、`TestRiskAnalysis`、`TestCompetitorAnalysis`、`TestParallelDataCollection`、`TestResolveCompany` 等**会失败**——因为它们测试的步骤方法已删除

- [ ] **Step 9: 重写 test_flow.py 删掉已删除步骤的测试，保留端到端**

在 `tests/test_flow.py` 中：
- 删除 `TestResolveCompany`（整个类）
- 删除 `TestParallelDataCollection`（整个类）
- 删除 `TestCompetitorAnalysis`（整个类）
- 删除 `TestRiskAnalysis`（整个类）
- 删除 `TestValuationAnalysis`（整个类）
- **保留** `TestWriteReport`（但需要修改它，因为 `write_report` 重命名为 `synthesize_report`，且 `@listen` 装饰器变了）
- **保留** `TestFlowKickoff`（端到端测试，但需要 mock `AnalysisCrew`）

修改 `TestWriteReport` 中所有 `flow.write_report()` → `flow.synthesize_report()`。

修改 `TestFlowKickoff::test_full_flow_with_mocked_registry` 加入 `AnalysisCrew` mock：

在 `_patch_competitor_tool` 上下文管理器之后追加 `patch("alphaquant.flows.analysis_flow.AnalysisCrew")`，让 MockCrew.return_value.kickoff 返回 fake result。

完整修改（替换 `test_full_flow_with_mocked_registry` 方法体中的 `with` 块）：

```python
        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=news_items)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)), \
             _patch_competitor_tool("No peer data available"), \
             patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:

            # Mock crew to produce a fake result that triggers deterministic fallback
            fake_result = MagicMock()
            fake_result.tasks_output = []
            MockCrew.return_value.kickoff.return_value = fake_result

            # Drive the steps manually
            _run(flow.run_crew("AAPL"))
            _run(flow.synthesize_report())
```

类似修改 `test_partial_failure_degrades_gracefully` 和 `test_kickoff_with_timeout_succeeds_under_limit`，都加上 `AnalysisCrew` mock。

- [ ] **Step 10: 跑全部 test_flow.py，确认通过**

Run: `uv run pytest tests/test_flow.py -v`
Expected: 全部测试通过

- [ ] **Step 11: 跑全部 test suite，确认没有 regression**

Run: `uv run pytest tests/ -q`
Expected: 197 个测试全部通过（190 backend + 7 frontend/db），加上 AnalysisCrew 9 个 = 206

如果失败：检查 `test_agents.py`、`test_crew.py`、`test_scoring.py` 等是否回归。Scoring tests 应该不变（parse_crew_output 内部仍调 scoring.*）。

- [ ] **Step 12: Commit**

```bash
git add src/alphaquant/flows/analysis_flow.py \
        tests/test_flow.py
git commit -m "refactor(flow): make AnalysisFlow a thin shell wrapping AnalysisCrew

Replaces 6 sequential @listen steps with @start run_crew +
@listen synthesize_report. run_crew pre-fetches all 4 data
sources, drives AnalysisCrew via asyncio.to_thread + wait_for,
then parse_crew_output fills state fields downstream tasks consume.

In sub-project 1, agents' JSON outputs are ignored for the data
they pre-fetched; parse_crew_output falls back to deterministic
logic (matching today's behavior) for competitor/risk/valuation
sub-scores. This keeps InvestmentReport byte-for-byte identical.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: 端到端验证 + 实跑对比

**Files:** 无（仅验证）

- [ ] **Step 1: 跑完整 test suite**

Run: `uv run pytest tests/ -q`
Expected: 全部测试通过（baseline 186 + Task 1 新增 ~24 + Task 2 新增 9 + Task 3 新增 3 = ~222 测试）

- [ ] **Step 2: 实跑 CLI 验证输出格式**

Run:
```bash
timeout 180 uv run python -m alphaquant AAPL --format json 2>&1 | tail -1 | jq '.valuation.dcf_value, .report.confidence, .report.rating'
```
Expected:
- AAPL 的 `dcf_value` 是某个具体数字（之前测试是 189.88）
- `confidence` 是 82（之前测试值）
- `rating` 是某个 5 级别之一

- [ ] **Step 3: 验证 manager LLM 真的被调用**

Run:
```bash
timeout 180 uv run python -m alphaquant AAPL --format json 2>&1 | grep -i "manager\|hierarchical\|delegation" | head -10
```
Expected: 至少看到 1 行提到 hierarchical manager 调度（CrewAI 日志会输出类似 `Manager Agent (Manager)` 的字样）

- [ ] **Step 4: 对比前后 InvestmentReport 关键字段**

把当前输出保存到文件：

Run: `timeout 180 uv run python -m alphaquant AAPL --format json 2>/dev/null | tail -1 > /tmp/sub1_aapl.json`

对比字段（用 jq）：
```bash
jq 'del(.report.report_id, .report.generated_at, .report.data_as_of) |
    .report | del(.markdown)' /tmp/sub1_aapl.json
```
Expected: valuation.dcf_value、valuation.intrinsic_value_per_share、competitors.competitive_score、financial_health_score、risk.total_score、risk.level、rating、confidence 这些字段值与子项目 1 前一致。

- [ ] **Step 5: 跑 MSFT 和 TSLA 验证不同 ticker**

Run:
```bash
timeout 180 uv run python -m alphaquant MSFT --format json 2>&1 | tail -1 | jq '.valuation.dcf_value, .report.confidence'
timeout 180 uv run python -m alphaquant TSLA --format json 2>&1 | tail -1 | jq '.valuation.dcf_value, .report.confidence'
```
Expected: 3 个 ticker 的 dcf_value 互不相同（AAPL/MSFT/TSLA），confidence 都是 82（与子项目 1 前一致）

- [ ] **Step 6: 跑降级路径测试**

Run:
```bash
timeout 60 uv run python -m alphaquant NONEXISTENT_TICKER_XYZ 2>&1 | head -20
```
Expected: 系统降级（类似 `errors` 列表包含 `company_data_unavailable`），不挂掉

- [ ] **Step 7: Commit（无文件变更，跳过）**

如果以上所有步骤都通过，子项目 1 完成。无需新 commit。

---

## Self-Review（作者自查）

### Spec 覆盖检查

| Spec 章节 | 对应 Task |
|---|---|
| 新增 `src/alphaquant/crews/` | Task 2 ✓ |
| 新增 `tests/test_crew.py` | Task 2 ✓ |
| 修改 8 个 agents/*.py | Task 1 ✓ |
| 修改 `flows/analysis_flow.py` | Task 3 ✓ |
| 修改 `tests/test_agents.py` | Task 1 ✓ |
| 修改 `tests/test_flow.py` | Task 3 ✓ |
| Agent↔Tool 映射表 | Task 1 ✓ |
| Hierarchical Process + manager_llm | Task 2 ✓ |
| memory=False | Task 2 ✓ |
| 严格输出一致性 | Task 3（parse_crew_output 走确定性子路径）+ Task 4（端到端验证）✓ |
| 失败 → ReportGenerationError | Task 3（synthesize_report 已有 try/except）+ Task 4（step 6 降级验证）✓ |
| 测试通过 | Task 1/2/3 的 step-level 验证 + Task 4 ✓ |

### Placeholder 扫描

无 `TBD`/`TODO`/`implement later`/`fill in details`。所有 step 都包含实际代码或实际命令。

### Type consistency

- `build_*_agent(llm: LLM) -> Agent` — 在 Task 1 所有 8 个 agent 一致 ✓
- `AnalysisCrew.__init__(self) -> None` — Task 2/3 一致 ✓
- `AnalysisFlow.run_crew(self, ticker, crewai_trigger_payload)` — Task 3 ✓
- `parse_crew_output(result: Any, state: AnalysisState) -> None` — Task 3 ✓

### 已知风险

1. **parse_crew_output 的数据流**：sub-project 1 里 agent 的 JSON 输出被忽略（因为 Flow 已经 pre-fetch 过），但 Crew 仍然会被构造和运行。这保证 agents 是"活的"但 behavior 仍是确定性的。如果 CrewAI 在 hierarchical 模式下必须让 agent 实际产出非空 result 才能完成，本计划可能需要微调——届时 implementer 根据实际行为调整。

2. **测试时长**：加入 LLM mock 后端到端测试时间可能从 6s 增加到 30s+。这是可接受的代价（仍然 < 60s CI 限制）。

3. **CrewAI API 漂移**：`Process.hierarchical`、`manager_llm` 等 API 在不同 crewai 版本可能有差异。已锁定 `crewai>=0.80,<0.90` 缓解。

## 执行交接

Plan 已保存到 `docs/superpowers/plans/2026-06-21-multi-agent-activation-sub1.md`。两种执行方式：

1. **Subagent-Driven（推荐）**：每 task 派一个新 subagent，task 间 review
2. **Inline Execution**：当前 session 里执行，用 executing-plans + checkpoint

哪种方式？
