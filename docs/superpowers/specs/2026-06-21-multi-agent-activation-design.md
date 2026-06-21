# AlphaQuant 多 Agent 激活 — 设计文档

> **范围**：本设计文档覆盖 4 个子项目中的 **子项目 1（Crew 编排壳）**。其他 3 个子项目（数据采集 agent、推理 agent、delegation/memory/retry）单独 spec。

## 背景

AlphaQuant 项目当前架构声称"8 个 CrewAI Agent 协同"，但实际上：
- `src/alphaquant/agents/*.py` 8 个 agent 文件全部是**死代码**——`build_*_agent()` 从未被调用过
- 生产路径是 100% 确定性 Python Flow（`flows/analysis_flow.py` 用 `@start`/`@listen` 直接调 `DataSourceRegistry` + `scoring.*`）
- 项目文档（`docs/ARCHITECTURE.md`）描述的"8 Agent + Flow"实际从未运行

用户确认系统**需要多 agent 协同**。本子项目（子项目 1）目标：把 8 个 agent 从死代码变成 CrewAI Crew 实际调度的对象，行为保持严格一致。

## 总体拆分（4 个子项目）

| # | 子项目 | 目标 | 预估 |
|---|---|---|---|
| 1 | **Crew 编排壳**（本文档） | `build_*_agent()` 不再死代码；Flow 退化为薄壳 | 2-3 天 |
| 2 | 数据采集 agent | 4 个数据 agent 真正在 Crew 里跑（从 Flow 移走） | 2-3 天 |
| 3 | 推理 agent 接 LLM | 4 个 reasoning agent 变 thick（有 LLM 推理） | 3-5 天 |
| 4 | Delegation + Memory + Retry/Degrade | 生产级多 agent 特性 | 3-5 天 |

实施顺序：1 → 2 → 3 → 4。

## 子项目 1 目标

**Goal**: 把 8 个 `build_*_agent()` 函数从死代码变成 CrewAI Crew 实际调度的对象；`AnalysisFlow` 退化为薄壳（只负责超时、状态解析、`InvestmentReport` 合成）。

**Non-goals**（明确不做）：
- 不改变 `InvestmentReport` 的任何字段值（byte-for-byte 一致）
- 不开 `allow_delegation=True`（sub-project 4 开启）
- 不开 CrewAI Memory（sub-project 4 开启）
- 不做 retry / degrade（sub-project 4 开启）
- 不让 agent 做真正的 LLM 推理（sub-project 3 开启）
- 不改 `core.py`、`scoring/*`、`infrastructure/*`、`models/*`

## 架构

### 整体

```
┌────────────────────────────────────────────────────────────────────┐
│  AnalysisFlow (CrewAI Flow, thin shell)                            │
│                                                                    │
│  @start async def run_crew(self, ticker)                           │
│    └─→ asyncio.to_thread(AnalysisCrew().kickoff(inputs))          │
│         └─→ asyncio.wait_for(timeout=120s)                         │
│    └─→ parse_crew_output(result) → fill self.state                 │
│                                                                    │
│  @listen(run_crew)                                                 │
│  async def synthesize_report(self)                                 │
│    └─→ build InvestmentReport from self.state (现有逻辑)            │
└────────────────────────────────────────────────────────────────────┘
                              │ invokes (sync, in thread)
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  AnalysisCrew (CrewAI Crew, hierarchical process)                  │
│                                                                    │
│  Crew(                                                              │
│    agents=[8 worker agents],                                       │
│    tasks=[8 tasks],                                                │
│    process=Process.hierarchical,                                   │
│    manager_llm=get_llm(temperature=0.1),                           │
│    memory=False,                                                   │
│    verbose=False,                                                  │
│  )                                                                  │
│                                                                    │
│  Manager agent: CrewAI 自动创建，按 task description 委派给 worker │
└────────────────────────────────────────────────────────────────────┘
```

### Agent ↔ Tool 映射（每个 agent 只装最直接相关的 tool）

| Agent | Tools | 说明 |
|---|---|---|
| CompanyResolver | (无) | 调 `DataSourceRegistry.get_company` |
| MarketAnalyst | `[MarketDataTool()]` | 调 MarketDataTool（实际仍调 `DataSourceRegistry.get_market`） |
| NewsAnalyst | `[NewsTool()]` | 调 NewsTool |
| FinancialAnalyst | `[FinancialTool()]` | 调 FinancialTool |
| CompetitorAnalyst | `[CompetitorTool()]` | 调 CompetitorTool |
| RiskAnalyst | (无) | 从 memory 读数据，调 `scoring.risk_score.compute` |
| ValuationAnalyst | `[DCFTool()]` | 调 DCFTool 拿默认假设 + `scoring.dcf.compute_dcf_value` |
| ReportWriter | (无) | 从 memory 读数据，调 `scoring.rating.determine_rating` + `_build_markdown` |

### Task 描述（8 个）

每个 task 用 `description="..."` 描述期望 agent 做什么，`expected_output="..."` 描述期望的输出 schema。子项目 1 里 task 描述保持极简（agent 实际工作仍是 Python 调用）：

```
1. "Resolve ticker to company metadata"
2. "Fetch market data for {ticker}"
3. "Fetch and analyze news for {ticker}"
4. "Fetch financial statements for {ticker}"
5. "Identify competitors and compute competitive score"
6. "Compute risk assessment"
7. "Compute valuation (DCF + relative + PEG)"
8. "Synthesize final InvestmentReport"
```

### Hierarchical Process 选择

子项目 1 用 `Process.hierarchical`（用户选定），尽管 worker agent 子项目 1 里几乎是 deterministic：
- **代价**：每次 analysis 多 8-15 次 manager LLM 调用（每次 1-2s + 极少 token）
- **好处**：子项目 4 开启 `allow_delegation=True` + memory 时**不需要重写 process 类型**，架构一次到位
- **缓解**：worker 行为确定性，manager 决策简单，总开销 < 5s

### Sync/Async 桥接

CrewAI `Crew.kickoff()` 是同步阻塞。`AnalysisFlow` 是 async。桥接方式：

```python
result = await asyncio.wait_for(
    asyncio.to_thread(crew.kickoff, inputs={"ticker": ticker}),
    timeout=FLOW_TIMEOUT_SECONDS,  # 120s
)
```

## 数据流

1. **入口**：`core.run_analysis_async(ticker)` → `AnalysisFlow.kickoff_with_timeout(inputs={"ticker": ticker})`
2. **`run_crew` 步骤**：
   - 构造 `AnalysisCrew` 实例
   - `crew.kickoff(inputs={"ticker": ticker})` → 阻塞运行 8 个 task（hierarchical）
   - Crew 返回 `CrewOutput` 对象（含 8 个 task 的输出 dict）
   - `parse_crew_output(result)` 把 dict 转成 `AnalysisState` 字段并填入 `self.state`
3. **`synthesize_report` 步骤**：
   - 从 `self.state` 构造 `InvestmentReport`（沿用现有 `_build_markdown`、`determine_rating`、`financial_health.compute` 等）
   - 写入 `self.state.report`

## 严格输出一致性

| `InvestmentReport` 字段 | 子项目 1 后来源 | 与现状差异 |
|---|---|---|
| `rating` / `confidence` | `scoring.rating.determine_rating()` | 0 |
| `valuation`（含 `dcf_value`） | `scoring.dcf.compute_dcf_value()` + relative | 0 |
| `risk` | `scoring.risk_score.compute()` | 0 |
| `competitors`（含 `competitive_score`） | `scoring.competitive.compute()` | 0 |
| `financial_health_score` | `scoring.financial_health.compute()` | 0 |
| `markdown` | `_build_markdown()` | 0 |
| `sources` | `_collect_sources()` | 0 |
| `catalysts` | `[]` | 0 |
| `errors`（Flow state 字段） | 现有逻辑 | 0 |

**保证机制**：每个 worker agent 的"工作"通过 **CrewAI 工具调用**完成（manager 委派 task → agent 收到 task → agent 调相关 tool 拿数据 → 输出结果）。已存在的 5 个工具（`MarketDataTool`、`NewsTool`、`FinancialTool`、`CompetitorTool`、`DCFTool`）都已封装对 `DataSourceRegistry` 或硬编码数据的调用，agent 直接复用它们即可。

LLM 在子项目 1 里只用于：
- Manager 委派决策（不修改任何字段）
- Agent 决定何时调哪个 tool 的最小推理（受 tool description 和 agent prompt 约束）

为了把 LLM 决策的影响降到最低：
- 每个 agent 的 `goal`/`backstory` 写得非常明确（"Use your tool to fetch X data; do not reason beyond that"）
- tool description 写得精确（让 LLM 知道什么时候该调）
- 子项目 1 的端到端测试需要验证：连续 3 次跑同一个 ticker，`InvestmentReport` 的非时间戳/UUID 字段必须完全一致

## 失败处理

子项目 1 暂时**不做 retry/degrade**（那是子项目 4）：
- Crew 任何异常 → 直接抛 `ReportGenerationError`
- FastAPI handler 按 spec §5.2 返回 500 INTERNAL_ERROR
- `errors` 列表里记录 `crew_failed: <message>`

## 文件变更清单

### 新增

| 路径 | 用途 |
|---|---|
| `src/alphaquant/crews/__init__.py` | package marker |
| `src/alphaquant/crews/analysis_crew.py` | `AnalysisCrew` 类（封装 CrewAI Crew + 8 Tasks） |
| `tests/test_crew.py` | AnalysisCrew 单元测试 |

### 修改

| 路径 | 变更 |
|---|---|
| `src/alphaquant/agents/*.py`（8 文件） | 每个 `build_*_agent()` 接收 `llm: LLM` 参数；返回的 `Agent.tools` 列表按上表配置 |
| `src/alphaquant/flows/analysis_flow.py` | 删除现有 6 个 `@listen` 步骤；新增 `@start run_crew` + `@listen synthesize_report` |
| `tests/test_flow.py` | 现有端到端测试改为 mock `AnalysisCrew` 让它返回固定 state |
| `tests/test_agents.py` | 改写为验证 `build_*_agent()` 返回真实 `Agent` 对象 + tools 配置正确 |

### 不动

```
src/alphaquant/scoring/                        # 纯函数继续作为 Python 模块被 agent 调用
src/alphaquant/infrastructure/                 # DataSourceRegistry、llm、config 全部不变
src/alphaquant/core.py                         # 仍然调 flow.kickoff_with_timeout()
src/alphaquant/models/                         # Pydantic schema 不变
src/alphaquant/observability/                  # 日志格式不变
src/alphaquant/tools/                          # 不变（仅被 agent 引用）
src/alphaquant/exceptions.py                   # 不变
src/alphaquant/main.py                         # 不变
```

## 测试计划

### 新增测试

```python
# tests/test_crew.py
class TestAnalysisCrew:
    def test_instantiates_without_error(self): ...
    def test_all_8_agents_built(self):
        assert len(crew.agents) == 8
        for agent in crew.agents:
            assert isinstance(agent, Agent)
    def test_all_8_tasks_built(self): ...
    def test_process_is_hierarchical(self):
        assert crew.crew.process == Process.hierarchical
    def test_manager_llm_configured(self):
        assert crew.crew.manager_llm is not None
    def test_tools_per_agent(self):
        # 表格化的 tools 映射校验
        expected = {
            "company_resolver": [],
            "market_analyst": [MarketDataTool],
            "news_analyst": [NewsTool],
            "financial_analyst": [FinancialTool],
            "competitor_analyst": [CompetitorTool],
            "risk_analyst": [],
            "valuation_analyst": [DCFTool],
            "report_writer": [],
        }
        for agent, (name, expected_tools) in zip(crew.agents, expected.items()):
            assert len(agent.tools) == len(expected_tools)
```

### 修改测试

```python
# tests/test_agents.py
class TestAgents:
    def test_company_resolver_has_no_tools(self):
        agent = build_company_resolver_agent(mock_llm)
        assert isinstance(agent, Agent)
        assert agent.tools == []
    # ... 8 个 agent × 3-4 个断言
```

```python
# tests/test_flow.py
class TestFlowWrapsCrew:
    def test_run_crew_invokes_analysis_crew(self, sample_market, ...):
        flow = AnalysisFlow()
        with patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:
            MockCrew.return_value.kickoff.return_value = fake_crew_output(...)
            _run(flow.run_crew("AAPL"))
            MockCrew.assert_called_once()
            MockCrew.return_value.kickoff.assert_called_once_with(inputs={"ticker": "AAPL"})
    def test_synthesize_report_from_crew_state(self, ...): ...
```

### 保留的测试

- `tests/test_scoring.py`（scoring 纯函数测试，不变）
- `tests/test_db.py`、`tests/test_observability.py` 等其他测试不变
- `tests/test_api.py`（API 集成测试，不变——只走 core.run_analysis_async）

### 验证清单

- [ ] `uv run pytest tests/ -q` 通过，新增至少 6 个 AnalysisCrew 测试
- [ ] `python -m alphaquant AAPL --format json` 输出和子项目 1 之前 byte-for-byte 一致（除了时间戳、UUID）
- [ ] `python -m alphaquant AAPL --format json` 的 timing log 显示有 manager agent 调度（hierarchical process 标志）
- [ ] 端到端测试通过率 100%（包括 `test_partial_failure_degrades_gracefully` 等 graceful 路径——crew 失败后仍走原 Flow 路径）

## 风险与权衡

1. **Hierarchical Process 的 manager LLM 开销**：每次 analysis 多 8-15 次 manager 决策 LLM 调用，每次 ~1-2s + 极少 token。worker 行为确定性 → manager 决策简单 → 总开销 < 5s。可接受。

2. **CrewAI 升级风险**：`crewai` 版本升级可能改变 `Crew`、`Process`、`Agent` API。需在 `pyproject.toml` 锁定版本（`crewai>=0.80,<0.90`）。

3. **测试时长增加**：端到端测试需要 mock LLM，单测时间可能从 6s 增加到 30s+。可通过更激进的 mock 缓解。

4. **`core.py` 兼容性**：完全不动，但实际响应时间变长（manager LLM 调用叠加）。需要在 README 加说明。

5. **CrewOutput 解析脆弱性**：`CrewOutput` 是 dict-like 对象，子项目 1 里通过 `parse_crew_output()` 函数手工解析。如果 CrewAI 升级改变输出格式，需要同步更新。需在测试里覆盖解析路径。

6. **state 与 memory 的边界**：子项目 1 不开 CrewAI memory，state 通过 `parse_crew_output()` 显式填充。子项目 4 开 memory 后，state ↔ memory 同步需要重新设计。

## 实施顺序

1. **Day 1**：
   - 新增 `src/alphaquant/crews/__init__.py` + `analysis_crew.py`（先不含 manager_llm，等 worker 跑通再加）
   - 修改 8 个 `agents/*.py` 接收 `llm` 参数 + 配置 tools
   - 写 `tests/test_agents.py` 验证

2. **Day 2**：
   - 改 `flows/analysis_flow.py`：`@start run_crew` + `@listen synthesize_report`
   - 写 `parse_crew_output()` 转换函数
   - 改 `tests/test_flow.py` 验证 Flow 包装 Crew

3. **Day 3**：
   - 加 `tests/test_crew.py` 完整覆盖
   - 跑完整 test suite 验证 regression
   - 实跑 `python -m alphaquant AAPL --format json` 对比前后输出

## 不在本子项目范围（明确划线）

- ❌ 让 worker agent 做真正的 LLM 推理 → 子项目 3
- ❌ 开启 `allow_delegation=True` → 子项目 4
- ❌ 开启 CrewAI Memory → 子项目 4
- ❌ 实现 retry / degrade → 子项目 4
- ❌ 拆分数据采集到独立 Crew task → 子项目 2
- ❌ 修改 `InvestmentReport` 任何字段语义 → 子项目 3+
- ❌ 删除 `tests/test_agents.py` → 重写而非删除
