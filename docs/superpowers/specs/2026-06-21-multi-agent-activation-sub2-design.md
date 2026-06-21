# AlphaQuant 多 Agent 激活 — Sub-Project 2 设计文档

> **范围**：本设计文档覆盖 4 个子项目中的 **子项目 2（数据 agent 真实运行）**。
> 子项目 1（Crew 编排壳）已完成（commits `7ff7af0`, `fd21a69`, `27a9ef4`, `42831e5`）。

## 背景

子项目 1 完成了 CrewAI Crew 编排壳，但 4 个数据 agent（CompanyResolver, MarketAnalyst, NewsAnalyst, FinancialAnalyst）实际上是**死的**：

- `AnalysisFlow.run_crew()` 在调用 crew 之前，先用 `asyncio.gather(...)` 同步预取 4 个数据源
- 数据作为 `crew.kickoff(inputs={"ticker": ..., "company": ..., "market": ..., "news": ..., "financial": ...})` 的输入喂给 crew
- Agents 收到 inputs 但**不调用自己的 tool**——因为数据已经在 state 里了
- `parse_crew_output` 对 4 个 data key 留 `pass` 空块

子项目 2 目标：**让 4 个数据 agent 真正在 Crew 里 fetch 数据**。Flow 只负责 orchestration + synthesis。

## 目标

**Goal**: 4 个数据 agent 通过自己的 tool 在 Crew 内 fetch 数据；Flow 完全删除预取；`InvestmentReport` 非时间戳/UUID 字段保持 byte-for-byte 一致。

**Non-goals**（明确不做）：
- 不让 agent 做真正的 LLM 推理（sub-3）
- 不开 `allow_delegation=True`（sub-4）
- 不开 CrewAI Memory（sub-4）
- 不做 retry / degrade（sub-4）
- 不改 `core.py`、`scoring/*`、`infrastructure/*`、`models/*`、`interfaces/*`
- 不改 `InvestmentReport` 任何字段语义

## 已确认的决策

1. Flow 完全删除预取（只剩 orchestration + synthesis）
2. Data agents 原样返回 tool 返回的 JSON（不做 LLM 总结）
3. 新增 `CompanyLookupTool`；`CompanyResolver.tools = [CompanyLookupTool()]`
4. 测试策略：Mock tools + `_FakeLLM`（与 sub-1 一致，零成本）
5. 4 data agents 并行跑（`Task(async_execution=True)`）

## 架构

### 数据流 before / after

```
Before (sub-1):
  run_crew:
    asyncio.gather(registry.get_company, get_market, get_news, get_financial)
    → state.company/market/news/financial 直接填
    crew.kickoff(inputs={company: ..., market: ..., news: ..., financial: ...})
    parse_crew_output: 抽取 competitor/risk/valuation（data 字段 Flow 已填）

After (sub-2):
  run_crew:
    crew.kickoff(inputs={"ticker": "AAPL"})  # 只传 ticker
      └─ Manager 调度 8 个 task；4 个 data task 标 async_execution=True 并行
        ├─ CompanyResolver → CompanyLookupTool → JSON
        ├─ MarketAnalyst   → MarketDataTool    → JSON
        ├─ NewsAnalyst     → NewsTool          → JSON list
        └─ FinancialAnalyst→ FinancialTool     → JSON
      └─ Competitor/Risk/Valuation/ReportWriter task 顺序（deterministic fallback）
    parse_crew_output:
      - 抽取 company/market/news/financial 从 4 个 data agent 的 task output
      - 抽取 competitor/risk/valuation（deterministic fallback 不变）
  synthesize_report: 不变
```

### Agent ↔ Tool 映射（更新）

| Agent | Tools | sub-2 变化 |
|---|---|---|
| CompanyResolver | `[CompanyLookupTool()]` | **+1 tool**（sub-1 为 `[]`） |
| MarketAnalyst | `[MarketDataTool()]` | + `asyncio.wait_for(30s)` 防 hang |
| NewsAnalyst | `[NewsTool()]` | + `asyncio.wait_for(30s)` |
| FinancialAnalyst | `[FinancialTool()]` | + `asyncio.wait_for(30s)` |
| CompetitorAnalyst | `[CompetitorTool()]` | 不变 |
| RiskAnalyst | `[]` | 不变 |
| ValuationAnalyst | `[DCFTool()]` | 不变 |
| ReportWriter | `[]` | 不变 |

### Task 描述（4 data task 加 `async_execution=True`）

```python
Task(
    description="Validate ticker '{ticker}' and return canonical company metadata.",
    expected_output="JSON with company name, exchange, sector, industry, market cap",
    agent=company_resolver_agent,
    async_execution=True,  # NEW in sub-2
)
```

并行执行由 CrewAI manager 在 hierarchical process 下调度（待 smoke test 验证 0.203.2 支持）。

## 数据流细节

### `parse_crew_output` 抽取 4 个 data 字段

每条 task output 是 string（tool 返回的 JSON 或 error message）。`parse_crew_output` 对每个 data key：

```python
def _extract_data_field(raw: str, model_cls, error_msg: str):
    """Try to parse JSON; if it fails or looks like error string, return None."""
    raw = raw.strip()
    if not raw or raw.startswith("Error") or raw.startswith("No ") or "data available" in raw:
        return None, error_msg
    try:
        return model_cls.model_validate_json(raw), None
    except ValidationError:
        return None, error_msg

# 调用方式：
elif key == "market_analyst":
    state.market, err = _extract_data_field(raw, MarketData, "market_data_unavailable")
    if err:
        state.errors.append(err)
elif key == "news_analyst":
    # NewsTool 返回 JSON list，要转 NewsAnalysis
    if raw looks like error:
        state.news = _empty_news_analysis(ticker)
        state.errors.append("news_data_unavailable")
    else:
        items = json.loads(raw)
        state.news = _news_items_to_analysis([NewsItem(**i) for i in items], ticker)
elif key == "financial_analyst":
    state.financial, err = _extract_data_field(raw, FinancialStatements, "financial_data_unavailable")
    if err:
        state.financial = FinancialStatements(ticker=normalized)  # 空壳
elif key == "company_resolver":
    company, err = _extract_data_field(raw, Company, "company_data_unavailable")
    if err:
        # 关键路径失败 → 立即抛 AllDataSourcesDown 保留错误码
        raise AllDataSourcesDown(f"No data source could resolve {normalized}")
    state.company = company
```

### `run_crew` 简化

```python
@start()
async def run_crew(self, ticker, crewai_trigger_payload=None) -> None:
    # ... ticker normalize 不变 ...

    # DELETE: asyncio.gather(registry.get_company, ...)
    # DELETE: state.company/market/news/financial 直接填

    crew = AnalysisCrew()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                crew.kickoff,
                inputs={"ticker": normalized},  # 只 ticker
            ),
            timeout=FLOW_TIMEOUT_SECONDS,  # 120s → 调整为 180s（并行 fetch + manager overhead）
        )
    except asyncio.TimeoutError:
        log.error("crew_timeout", ticker=normalized)
        raise

    parse_crew_output(result, self.state)  # 现在填充 4 个 data 字段 + competitor/risk/valuation
```

### `_news_items_to_analysis` 调用方变化

- **sub-1**: Flow 在 `run_crew` 里调用 `_news_items_to_analysis(news_list, ticker)` 把 `list[NewsItem]` 转 `NewsAnalysis`
- **sub-2**: `_news_items_to_analysis` 移到 `parse_crew_output` 的 `news_analyst` 分支里

## 失败处理

| 失败模式 | sub-1 处理 | sub-2 处理 |
|---|---|---|
| Tool 抛异常 | Flow `gather(return_exceptions=True)` 兜住 | Tool `_run` 内 `try/except` 转 error string |
| Tool 30s timeout | Flow `wait_for(45s)` 兜 4 个 | Tool `_run` 内 `wait_for(30s)` 兜单次 |
| Tool 返回 None/empty | Flow 设 None + 加 error | Tool 返回 error string，`parse_crew_output` 识别 → None + error |
| Crew 整体 hang | `wait_for(120s)` | `wait_for(180s)`（4 并行 + manager overhead） |
| Company fetch 失败 | Flow 入口抛 `AllDataSourcesDown` | `parse_crew_output` 检测 company_resolver task output 失败 → 抛 `AllDataSourcesDown`（保留错误码） |
| 其他 3 个 data fetch 失败 | Flow 设 None + 加 error | `parse_crew_output` 设 None/空壳 + 加 error |
| Manager LLM 失败 | `litellm.BadRequestError` | 同（pre-existing limitation 与 API key 有关） |

**关键调整**：`FLOW_TIMEOUT_SECONDS` 从 120s 调整为 180s。3 个并行 data fetch（每个 30s 内）+ 4 个 manager decision（每个 ~2s）+ 兜底 ≈ 150-180s。

## 严格输出一致性

| `InvestmentReport` 字段 | sub-1 来源 | sub-2 来源 | 差异 |
|---|---|---|---|
| `company.*` | `registry.get_company()` | CompanyResolver → CompanyLookupTool → `registry.get_company()` | 0 |
| `market.*` | `registry.get_market()` | MarketAnalyst → MarketDataTool → `registry.get_market()` | 0 |
| `news.*` | `_news_items_to_analysis(registry.get_news(), ticker)` | NewsAnalyst → NewsTool → `_news_items_to_analysis(...)` | 0 |
| `financial.*` | `registry.get_financial()` | FinancialAnalyst → FinancialTool → `registry.get_financial()` | 0 |
| `competitor`, `risk`, `valuation` | `parse_crew_output` deterministic fallback | 同（不变） | 0 |
| 时间戳（`market.as_of`, `news.as_of`, `report.generated_at`） | 每次 fetch 新鲜 | 同 | 不保证 byte-for-byte（本来就不） |

**保证机制**：4 个数据 agent 的 tool 直接复用 `DataSourceRegistry`，调用相同的底层 source（Yahoo → AlphaVantage → ...），返回相同模型（Pydantic），最终由 `parse_crew_output` 用相同的 `_news_items_to_analysis` 等函数转换。

## 文件变更清单

### 新增

| 路径 | 用途 |
|---|---|
| `src/alphaquant/tools/company_lookup_tool.py` | `CompanyLookupTool(BaseTool)` 包装 `registry.get_company()` |

### 修改

| 路径 | 变更 |
|---|---|
| `src/alphaquant/agents/company_resolver.py` | `tools=[CompanyLookupTool()]` |
| `src/alphaquant/tools/market_data_tool.py` | 加 `asyncio.wait_for(timeout=30)` |
| `src/alphaquant/tools/financial_tool.py` | 同上 |
| `src/alphaquant/tools/news_tool.py` | 同上 |
| `src/alphaquant/crews/analysis_crew.py` | 4 data task 加 `async_execution=True` |
| `src/alphaquant/flows/analysis_flow.py::run_crew` | 删 4 个 registry 调用；`kickoff(inputs={"ticker": normalized})` |
| `src/alphaquant/flows/analysis_flow.py::parse_crew_output` | 4 个 data 字段从 agent output 抽取 |
| `src/alphaquant/flows/analysis_flow.py` | `FLOW_TIMEOUT_SECONDS` 120 → 180 |
| `tests/test_agents.py` | CompanyResolver test 改：1 tool, CompanyLookupTool |
| `tests/test_flow.py::TestRunCrewStep` | 不再 mock DataSourceRegistry；改 mock 4 tool `_run` |
| `tests/test_flow.py::TestParseCrewOutput` | 加 4 个 data 字段抽取测试 |
| `tests/test_tools.py` | 加 `TestCompanyLookupTool` |

### 不动

```
src/alphaquant/scoring/                        # scoring 纯函数不变
src/alphaquant/infrastructure/data_sources/    # DataSourceRegistry 不变
src/alphaquant/infrastructure/llm.py           # get_llm 不变
src/alphaquant/infrastructure/config.py        # Settings 不变
src/alphaquant/models/                         # Pydantic schema 不变
src/alphaquant/observability/                  # 日志格式不变
src/alphaquant/exceptions.py                   # AllDataSourcesDown 不变
src/alphaquant/main.py                         # 不变
src/alphaquant/core.py                         # 不变
src/alphaquant/interfaces/cli.py               # 不变
src/alphaquant/interfaces/api/                 # 不变
src/alphaquant/interfaces/frontend/            # 不变
src/alphaquant/agents/{market,news,financial,competitor,risk,valuation,report_writer}.py  # backstory 微调提到 tool 名（让 LLM 知道调哪个），结构不变
```

## 测试计划

### 新增测试

```python
# tests/test_tools.py — 新增 TestCompanyLookupTool
class TestCompanyLookupTool:
    def test_returns_company_json_on_success(self, monkeypatch):
        """Mock registry.get_company → Company instance; tool returns its JSON."""

    def test_returns_error_string_on_exception(self, monkeypatch):
        """Mock registry raising; tool returns 'Error: ...' string."""

    def test_timeout_returns_error_string(self, monkeypatch):
        """Mock registry hanging; tool returns timeout error string."""
```

### 修改测试

```python
# tests/test_agents.py
def test_company_resolver_has_company_lookup_tool(fake_llm):
    agent = build_company_resolver_agent(fake_llm)
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], CompanyLookupTool)

# tests/test_flow.py::TestRunCrewStep
def test_run_crew_does_not_pre_fetch(sample_company, sample_market, ...):
    """run_crew 不应调用 DataSourceRegistry；改 mock 4 个 tool 的 _run。"""

def test_run_crew_only_passes_ticker_to_crew(self, ...):
    """kickoff inputs 字典只有 ticker key。"""

# tests/test_flow.py::TestParseCrewOutput — 新增
def test_extracts_company_from_task_output()
def test_extracts_market_from_task_output()
def test_extracts_news_from_task_output()
def test_extracts_financial_from_task_output()
def test_company_fetch_failure_raises_all_sources_down()
def test_market_fetch_failure_sets_none_and_appends_error()
def test_news_fetch_failure_uses_empty_news_analysis()
def test_financial_fetch_failure_uses_empty_shell()
def test_tool_error_string_recognized_as_failure()
def test_valid_json_recognized_as_success()
```

### 保留的测试

- `tests/test_scoring.py`（scoring 纯函数不变）
- `tests/test_db.py`, `tests/test_observability.py`, `tests/test_tools.py` 已有部分
- `tests/test_api.py`（端到端不变）
- `tests/test_agents.py` 已有（除 CompanyResolver test）
- `tests/test_crew.py`（AnalysisCrew tests 不变 — 但要更新 `test_tools_mapping` 加 CompanyLookupTool 期望）

### 验证清单

- [ ] `uv run pytest tests/ -q` 通过，新增至少 6 个 test，原 204 不回归
- [ ] `python -m alphaquant AAPL --format json` 输出 `dcf_value` 与 sub-1 相同
- [ ] 实跑 3 个 ticker (AAPL/MSFT/TSLA) 的 dcf_value 仍然互不相同
- [ ] graceful degradation 路径仍然工作（INVALID_TICKER_FORMAT, ALL_DATA_SOURCES_DOWN）
- [ ] Flow 代码里没有 `DataSourceRegistry.get_*` 调用（grep 验证）

## 风险与权衡

1. **CrewAI 0.203.2 是否在 hierarchical process 下支持 `Task(async_execution=True)`？** Smoke test 验证。如果不支持，回退到顺序（接受 ~20s 总耗时）。

2. **Sequential → parallel 性能**：sub-1 的 `asyncio.gather` 并行 fetch 4 个数据源 ~5s。sub-2 用 CrewAI 4 task 并行可能略慢（每个 task 启动 + manager delegation ~1s）。估算总耗时从 ~10s 到 ~15-25s。可接受。

3. **Tool error string vs JSON 判别**：当前设计以 `"Error"` / `"No "` / `"data available"` 字符串前缀识别。Agent 可能输出含这些关键词的合法 JSON。Mitigation：先 `model_validate_json`，失败再字符串判别。

4. **`AllDataSourcesDown` 时机**：sub-1 在 Flow 入口同步抛；sub-2 在 `parse_crew_output` 里 company_resolver 分支检测后抛。错误码保留，FastAPI handler 行为不变。

5. **`FLOW_TIMEOUT_SECONDS` 120 → 180**：3 并行 fetch（各 30s）+ 4 manager decisions（各 ~2s）+ 兜底 ≈ 150-180s。180s 留 30s 缓冲。

6. **`MINIMAX_API_KEY` 占位 → 真实**：用户已填入真实 key，sub-1 端到端测试现在可以跑。sub-2 实跑验证可用。

## 实施顺序（4 tasks，预计 2-3 天）

1. **Day 1**：
   - 新增 `CompanyLookupTool` + `tests/test_tools.py` 测试
   - 修改 4 个 tool 加 `asyncio.wait_for(timeout=30)`
   - 修改 `CompanyResolver` 加 tool
   - 改 `test_agents.py` 期望新 tool

2. **Day 2**：
   - 修改 `crews/analysis_crew.py`：4 data task 加 `async_execution=True`
   - 修改 `flows/analysis_flow.py::run_crew`：删预取、`kickoff(inputs={"ticker": ...})`、调整 `FLOW_TIMEOUT_SECONDS`
   - 修改 `flows/analysis_flow.py::parse_crew_output`：4 data 字段抽取 + `_extract_data_field` helper
   - 改 `tests/test_flow.py::TestRunCrewStep` 和 `TestParseCrewOutput`

3. **Day 3**：
   - 跑完整 test suite，验证 210+ pass
   - 实跑 AAPL/MSFT/TSLA 验证 `dcf_value` 与 sub-1 一致
   - 实跑 NONEXISTENT_TICKER 验证 graceful degradation

## 不在 sub-2 范围（明确划线）

- ❌ Agent 真正的 LLM 推理 → sub-3
- ❌ 开启 `allow_delegation=True` → sub-4
- ❌ 开启 CrewAI Memory → sub-4
- ❌ 实现 retry / degrade → sub-4
- ❌ 修改 `InvestmentReport` 任何字段语义
- ❌ 重构 `AnalysisCrew` 之外的 crew 行为（task 顺序、manager delegation）
- ❌ 修改非 data agent 的 tools 配置（Competitor/Risk/Valuation/ReportWriter 不变）
