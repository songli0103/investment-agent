# AlphaQuant 多 Agent 激活 — Sub-Project 3 设计文档

> **范围**：本设计文档覆盖 4 个子项目中的 **子项目 3（推理 agent 真实 LLM 推理 + Pydantic 结构化输出）**。
> 子项目 1（Crew 编排壳）已完成（commit `42831e5`）；子项目 2（数据 agent 真实运行）已完成但验证未通过（commits `a26b9c9, da506da, 8adb8b7, f404719, 632d96b`）。

## 背景

子项目 2 完成后，4 个数据 agent（CompanyResolver, MarketAnalyst, NewsAnalyst, FinancialAnalyst）已经在 Crew 内通过自己的 tool 抓取数据，`parse_crew_output` 也扩展到抽取这 4 个 data 字段。但 4 个 analysis agent（CompetitorAnalyst, RiskAnalyst, ValuationAnalyst, ReportWriter）**仍然只输出原始文本，Flow 用 deterministic fallback 覆盖它们的输出**：

- `_populate_competitor()` 用 Flow-side `scoring.competitive.compute()` 算 competitive_score
- `_populate_risk()` 用 Flow-side `_default_risk_subscores()` + `scoring.risk_score.compute()` 算 risk
- `_populate_valuation()` 用 Flow-side `scoring.dcf.compute_dcf_value()` 算 DCF（hardcoded `peer_pe_avg=20.0`）
- `synthesize_report()` 用 Flow-side `scoring.rating.determine_rating()` 算 rating + confidence

后果：
- `InvestmentReport.rating` / `confidence` / `catalysts` / `investment_horizon` 是公式/硬编码的，不是 LLM 推理
- AAPL/MSFT/TSLA 跑出来 confidence 总是 82（"happy path" 公式固定解）—— 根因 = 0.4×data_completeness + 0.2×method_coverage + 0.4×signal_alignment，LLM 没参与
- 子项目 2 端到端验证还留下 **3 个 deferred blocker**：
  1. `parse_crew_output` 报 `RuntimeError: cannot schedule new futures after shutdown`（asyncio shutdown race）
  2. `FLOW_TIMEOUT_SECONDS=180` 对真实 LLM 太短（MSFT/TSLA 在 risk rationale 阶段超时）
  3. 4 个 data tool 在 unknown ticker 时返回 `{"price": "N/A", ...}` 空壳而不是 `AllDataSourcesDown`，spec 的 graceful-degradation 路径端到端走不通

子项目 3 目标：**让 4 个 analysis agent 真的做 LLM 推理并产出 Pydantic 结构化输出，移除所有 deterministic fallback；同时修复 3 个 deferred blocker**。

## 目标

**Goal**: 4 个 analysis agent 通过 `Task(output_pydantic=...)` 产出 Pydantic 严格结构化输出；Flow 不再覆盖 agent 输出；ReportWriter agent 直接产出完整 `InvestmentReport`（含 rating/confidence/catalysts/investment_horizon/markdown）；3 个 deferred blocker 全部修复。

**Non-goals**（明确不做）：
- 不开 `allow_delegation=True`（sub-4）
- 不开 CrewAI Memory（sub-4）
- 不做 retry / degrade（sub-4）
- 不改 `core.py`、`interfaces/cli.py`、`interfaces/api/`、`interfaces/frontend/`、`infrastructure/*`（除 llm timeout 相关）
- 不改 `InvestmentReport` Pydantic schema 字段（但 LLM 现在驱动这些字段的值）
- 不改 `models/company.py`、`models/financial.py`、`models/market.py`、`models/news.py`（4 个 data agent 用的模型不动）
- 不改 4 个 data agent（sub-2 已完成）

## 已确认的决策

1. **Pydantic 路径**：`Task(output_pydantic=...)`，依赖 CrewAI 0.203.2 + MiniMax-M3 产出 Pydantic-valid JSON
2. **Fallback**：**完全删除** `_populate_competitor` / `_populate_risk` / `_populate_valuation` deterministic fallback。LLM 失败 → `state.field = None` + `errors.append(...)`
3. **LLM 决定字段**：`InvestmentReport.rating` / `confidence` / `catalysts` / `investment_horizon` / `markdown` 全部由 ReportWriter agent 决定（不再走 `scoring.rating.determine_rating()` 公式）
4. **执行顺序**：Competitor/Risk/Valuation 3 个并行（`async_execution=True`），ReportWriter 串行在它们之后，ReportWriter 用 `context=[task_4, task_5, task_6]` 拿前 3 者 Pydantic 输出
5. **scoring 拆分**：保留 `scoring/dcf.py`（ValuationAnalyst agent 仍调）、`scoring/financial_health.py`（ReportWriter 仍调作为 tool 算出 financial_health_score）。删除 `scoring/rating.py`、`scoring/competitive.py`、`scoring/risk_score.py`
6. **3 个 blocker 修复策略**（详见"修复 3 个 Deferred Blocker"节）：
   - Blocker 1: `crew.kickoff` 改同步调用（无 `asyncio.to_thread`），外层 `asyncio.wait_for(timeout)` 只管 timeout
   - Blocker 2: `FLOW_TIMEOUT_SECONDS` 180 → 600
   - Blocker 3: tool 收到 `AllDataSourcesDown` 不再 fallback 到空壳，把异常透传给 agent

## 架构

### 数据流 before / after

```
Before (sub-2):
  run_crew:
    asyncio.to_thread(crew.kickoff, inputs={"ticker": ...})
      └─ 8 tasks: 4 data (async) + 4 analysis (sequential)
    parse_crew_output:
      - 4 data fields from agent task outputs (raw tool JSON)
      - 3 analysis fields IGNORED, Flow 算 (deterministic)
    synthesize_report:
      - scoring.rating.determine_rating() → rating + confidence (公式)
      - 拼装 InvestmentReport

After (sub-3):
  run_crew:
    crew.kickoff(inputs={"ticker": ...})  # 同步
      └─ 8 tasks: 4 data (async) + 3 analysis (async) + 1 report (sequential)
    parse_crew_output:
      - 4 data fields from agent task outputs (unchanged from sub-2)
      - 3 analysis fields from output_pydantic (Pydantic model)
      - 1 report field from output_pydantic=InvestmentReport
    synthesize_report:
      - 直接用 state.report (ReportWriter agent 已输出完整 InvestmentReport)
      - 不再算 rating/confidence，不再调 scoring.rating
```

### Task 模板变化

```python
# crews/analysis_crew.py
_TASK_TEMPLATES: list[tuple[str, str, type[BaseModel] | None]] = [
    # (role_key, description_template, output_pydantic_model)
    ("company_resolver",  "Validate ticker '{ticker}' and return canonical company metadata.", None),
    ("market_analyst",    "Fetch market data for '{ticker}'.", None),
    ("news_analyst",      "Fetch recent news for '{ticker}'.", None),
    ("financial_analyst", "Fetch financial statements for '{ticker}'.", None),
    ("competitor_analyst","Identify competitors and compute competitive score for '{ticker}'.", CompetitorAnalysis),
    ("risk_analyst",      "Compute risk assessment for '{ticker}' from upstream data.",        RiskAssessment),
    ("valuation_analyst", "Compute valuation (DCF + relative) for '{ticker}'.",                ValuationResult),
    ("report_writer",     "Synthesize InvestmentReport for '{ticker}'.",                       InvestmentReport),
]

# ReportWriter (idx 7) 串行，依赖 task 4/5/6
_REPORT_WRITER_INDEX = 7
_ASYNC_TASK_INDICES = {0, 1, 2, 3, 4, 5, 6}  # 含 analysis agents 并行

# report_writer task 构造时传 context 参数
Task(
    description=...,
    expected_output="...",  # 仍写字符串，但 Pydantic 实际产出
    agent=report_writer_agent,
    context=[self.tasks[i] for i in (4, 5, 6)],  # 拿前 3 个 analysis 的 Pydantic 输出
    async_execution=False,
)
```

### Agent ↔ Tool 映射（更新）

| Agent | Tools | sub-3 变化 |
|---|---|---|
| CompanyResolver | `[CompanyLookupTool()]` | **不变**（sub-2） |
| MarketAnalyst | `[MarketDataTool()]` | **不变** |
| NewsAnalyst | `[NewsTool()]` | **不变** |
| FinancialAnalyst | `[FinancialTool()]` | **不变** |
| CompetitorAnalyst | `[CompetitorTool()]` | **+ backtory 重写** 强调 Pydantic 必填字段；`output_pydantic=CompetitorAnalysis` |
| RiskAnalyst | `[]` | **+ backtory 重写** 强调 6 个 RiskScore 类别全填；`output_pydantic=RiskAssessment` |
| ValuationAnalyst | `[DCFTool()]` | **+ backtory 重写**；`output_pydantic=ValuationResult`（含 dcf_value, intrinsic_value_per_share, current_price, upside_pct, assumptions） |
| ReportWriter | `[]` | **+ backtory 重写**；`output_pydantic=InvestmentReport`（含 rating, confidence, catalysts, investment_horizon, markdown, sources, disclaimer） |

### Pydantic schema 一致性

`output_pydantic` 直接用现有 Pydantic 模型（已 strict）：
- `CompetitorAnalysis`（`models/competitor.py`）：含 `target_ticker, competitors, industry_rank, industry_size, competitive_score, strengths, weaknesses, method`
- `RiskAssessment`（`models/risk.py`）：含 `ticker, total_score, level, sub_scores (≥1), top_risks (≤5), method="weighted_sum_v1"`
- `ValuationResult`（`models/valuation.py`）：含 `ticker, intrinsic_value_per_share, current_price, upside_pct, dcf_value, relative_value, peg_ratio, method Literal["dcf_relative_peg","relative_only"]`
- `InvestmentReport`（`models/report.py`）：含 `report_id, ticker, generated_at, data_as_of, company, market, financial, financial_health_score, news, competitors, risk, valuation, rating, confidence, investment_horizon, catalysts, markdown, sources, disclaimer`

**调整**：如果 sub-3 plan 阶段发现 LLM 倾向产出 `ValuationResult.method` 不在 `dcf_relative_peg|relative_only` 范围内（例如 `dcf_only`、`relative_only_with_manual_assumptions`、`llm_estimate`），plan 会把 Literal 扩到包含所有合理值。这是允许范围内的 schema 扩展，不改变字段语义（仅扩枚举值）。

## 数据流细节

### `parse_crew_output` 简化

```python
def parse_crew_output(
    result: CrewOutput, state: "AnalysisState"
) -> dict[str, Any]:
    tasks_output = result.tasks_output or []
    extracted: dict[str, Any] = {}

    # Data fields (sub-2 unchanged)
    _parse_data_field(tasks_output, 0, "company_resolver",  Company, state, raise_on_fail=True)
    _parse_data_field(tasks_output, 1, "market_analyst",    MarketData, state)
    _parse_news_field(tasks_output, 2, "news_analyst",      state)
    _parse_data_field(tasks_output, 3, "financial_analyst", FinancialStatements, state)

    # Analysis fields (NEW in sub-3): direct Pydantic extraction
    state.competitor = _extract_pydantic_field(
        tasks_output, 4, "competitor_analyst", CompetitorAnalysis, state
    )
    state.risk = _extract_pydantic_field(
        tasks_output, 5, "risk_analyst", RiskAssessment, state
    )
    state.valuation = _extract_pydantic_field(
        tasks_output, 6, "valuation_analyst", ValuationResult, state
    )

    # ReportWriter (NEW in sub-3): direct InvestmentReport
    state.report = _extract_pydantic_field(
        tasks_output, 7, "report_writer", InvestmentReport, state
    )

    return extracted


def _extract_pydantic_field(
    tasks_output: list, idx: int, key: str,
    model_cls: type[BaseModel], state: "AnalysisState"
) -> BaseModel | None:
    """从 task_output 取 Pydantic 字段。失败 → None + error。"""
    if idx >= len(tasks_output):
        state.errors.append(f"{key}_unavailable")
        return None
    task_out = tasks_output[idx]

    # CrewAI Task(output_pydantic=...) 成功后 task_out.pydantic 直接是 model 实例
    pyd_obj = getattr(task_out, "pydantic", None)
    if isinstance(pyd_obj, model_cls):
        return pyd_obj

    # fallback: pydantic_output / raw 字段（不同 CrewAI 版本路径不同）
    raw = getattr(task_out, "raw", "") or ""
    if not raw or raw.startswith("Error") or raw.startswith("No "):
        state.errors.append(f"{key}_unavailable")
        return None
    try:
        return model_cls.model_validate_json(raw)
    except ValidationError:
        state.errors.append(f"{key}_parse_failed")
        return None
```

### `synthesize_report` 简化

```python
@listen(run_crew)
def synthesize_report(self) -> None:
    """Sub-3: state.report 由 ReportWriter agent 直接产出，本步只做最终校验。"""
    if self.state.report is None:
        # ReportWriter 失败 → 走 INTERNAL_ERROR 兜底
        raise ReportGenerationError("Report writer agent failed to produce InvestmentReport")

    # sources 字段根据上游 state 重新汇总（sub-2 既有逻辑保留）
    self.state.report.sources = _collect_sources(
        self.state.market, self.state.news, self.state.financial, self.state.competitor
    )

    # disclaimer / generated_at 等运行时字段（sub-2 既有逻辑保留）
    self.state.report.disclaimer = DISCLAIMER_TEXT
    self.state.report.generated_at = datetime.utcnow()
```

### Agent backtory 重写示例

```python
# agents/risk_analyst.py
def build_risk_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Risk Assessment Specialist",
        goal=(
            "Compute risk assessment from upstream data. Output a RiskAssessment "
            "with sub-scores for ALL 6 categories: financial, operational, market, "
            "regulatory, governance, macro. Each sub-score must have rationale ≥10 chars."
        ),
        backstory=(
            "You are a senior risk officer. You MUST output a Pydantic RiskAssessment "
            "object. You are forbidden from omitting any of the 6 risk categories. "
            "Each rationale must be ≥10 characters and reference specific data from "
            "the financial statements or market data."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
```

## 修复 3 个 Deferred Blocker

### Blocker 1: `parse_crew_output` asyncio shutdown race

**根因**（sub-2 报告中的假设，sub-3 plan 阶段需要先复现/确认）：`run_crew` 用 `asyncio.to_thread(crew.kickoff, inputs=...)` 在 worker thread 跑 crew。`crew.kickoff` 内部会用 CrewAI 自己的 executor / event loop。worker thread 完成后回到主 event loop，但某些 shared state（executor 池）可能已被 FastAPI / CLI 框架 teardown，导致 `parse_crew_output` 内部任何 `await` 都报 "cannot schedule new futures after shutdown"。

**修复**：
- 把 `crew.kickoff` 调用包在 sync 函数 `_kickoff_sync()` 里，外层 `asyncio.to_thread(_kickoff_sync)` 跑在 worker thread
- 但 `asyncio.wait_for` 围在外面，让 timeout 触发后能 cancel worker thread 的 future
- `parse_crew_output` 改纯同步（已经是了，**只**去掉任何隐式 `await`）
- 与 sub-2 的差异：sub-2 直接 `asyncio.to_thread(crew.kickoff, ...)`，把 `crew.kickoff` 这个 method 引用当 callable。sub-3 改成把 `crew.kickoff(inputs=...)` 的实际调用挪进 sync 函数，确保 `wait_for` 的 cancel 能真正作用在 crew 的执行路径上（而不是只 cancel 了一个已开始返回的 future）

```python
@start()
async def run_crew(self) -> None:
    # ... normalize ticker ...
    def _kickoff_sync() -> CrewOutput:
        # 把 crew.kickoff 的实际调用挪进 sync 函数
        return AnalysisCrew().kickoff(inputs={"ticker": normalized})

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_kickoff_sync),  # wait_for 可 cancel 这个 future
            timeout=FLOW_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.error("crew_timeout", ticker=normalized)
        raise
    parse_crew_output(result, self.state)  # 纯同步
```

**注意**：上面的 `asyncio.to_thread(_kickoff_sync)` 是为了"timeout 触发后能 cancel worker thread"。如果 crew 内部不接受 cancellation 而 hang，仍会卡到 thread pool 关闭。Mitigation：设 `FLOW_TIMEOUT_SECONDS=600`，让"超时"是 pathological case。

### Blocker 2: `FLOW_TIMEOUT_SECONDS=180` 太短

**修复**：180 → 600。

新耗时估算：
- 4 个 data task 并行：~30s（最长 30s + manager overhead ~5s）
- 3 个 analysis task 并行：~60-90s（每个 LLM call ~20-30s，含 thinking）
- 1 个 ReportWriter task：~30-60s
- 总和：~120-180s。600s 留 3-4× 缓冲。

### Blocker 3: tool 返回空壳而非 `AllDataSourcesDown`

**现状**（sub-2）：`tools/company_lookup_tool.py` 收到 `AllDataSourcesDown` 后 fallback 返回 `{"name": "N/A", ...}` 空壳。`tools/market_data_tool.py` 同理。

**修复**：tool 在 `AllDataSourcesDown` 时返回 error string `"Cannot resolve {ticker}: company data unavailable"`（保留 sub-2 已有的 `Error fetching X: ...` 约定），**不** fallback 到空壳。`parse_crew_output` 已经会检测到 error string 并 raise `AllDataSourcesDown`（sub-2 已有逻辑）。

具体改动：
```python
# tools/company_lookup_tool.py::_run
try:
    company = loop.run_until_complete(asyncio.wait_for(
        registry.get_company(ticker), timeout=TOOL_TIMEOUT_SECONDS
    ))
except AllDataSourcesDown as e:
    return f"Error fetching company: {e}"  # 已经是这样
except Exception as e:
    # 之前这里 fallback 到 {"name": "N/A", ...} —— 删掉
    return f"Error fetching company: {type(e).__name__}: {e}"
```

**保留 sub-2 的 30s timeout 行为**，但**去掉** `except Exception` 里的空壳 fallback。empty dict fallback 只在"tool 完全没被调用 / agent 完全没产出" 时被使用（即 task output 为空字符串），由 `parse_crew_output._extract_data_field` 的 `if not raw` 分支处理。

## 失败处理

| 失败模式 | sub-2 处理 | sub-3 处理 |
|---|---|---|
| Tool 抛 `AllDataSourcesDown` | sub-2 已转 error string | **同上**（保留） |
| Tool 30s timeout | sub-2 已转 error string | **同上** |
| Tool 抛其他 Exception | sub-2 fallback 到空壳 dict | **删除空壳 fallback**；返回 error string |
| 4 个 data agent 失败 | sub-2 `parse_crew_output` 检测后 raise / 加 error | **同上**（保留） |
| 3 个 analysis agent 失败 | sub-2 `parse_crew_output` 用 Flow 算 | **删除 Flow 算**；`state.competitor/risk/valuation = None` + `errors.append("xxx_unavailable")` |
| ReportWriter agent 失败 | sub-2 Flow 拼装 | **`synthesize_report` raise `ReportGenerationError`** |
| Crew hang | `wait_for(180s)` | `wait_for(600s)` |
| LLM 产出 Pydantic 失败 | sub-2 不发生（不用 Pydantic） | `parse_crew_output._extract_pydantic_field` 检测后 `state.field = None` + error |

## 严格输出一致性

`InvestmentReport` 非时间戳/UUID 字段不再 byte-for-byte 相同 —— sub-3 起这些字段由 LLM 决定，confidence / rating / catalysts / markdown 每次跑都不同。

**保留** byte-for-byte 一致性的字段：
- `company.*` / `market.*` / `news.*` / `financial.*`（仍走 4 个 data agent 抓数据）
- `competitors.competitors` 列表（仍由 CompetitorAnalyst 抓的 peers 组成）

**LLM-决定** 字段（sub-3 起每次跑都不同）：
- `rating`, `confidence`, `investment_horizon`, `catalysts`, `markdown`, `sources`（汇总）, `disclaimer`（写死中文）

## 文件变更清单

### 修改

| 路径 | 变更 |
|---|---|
| `src/alphaquant/crews/analysis_crew.py` | `_TASK_TEMPLATES` 改为 `(key, desc, pydantic_model)` tuple；`_ASYNC_TASK_INDICES = {0..6}`；report_writer task 加 `context=[tasks[4..6]]`；删 `_TASK_KEYWORDS`（不再需要） |
| `src/alphaquant/agents/competitor_analyst.py` | backtory 重写强调 Pydantic 必填 |
| `src/alphaquant/agents/risk_analyst.py` | backtory 重写强调 6 类 RiskScore 全填 |
| `src/alphaquant/agents/valuation_analyst.py` | backtory 重写强调 ValuationResult 必填 |
| `src/alphaquant/agents/report_writer.py` | backtory 重写强调 InvestmentReport 全部字段 |
| `src/alphaquant/flows/analysis_flow.py` | `parse_crew_output` 重写为 `_extract_pydantic_field` 路径；删 `_populate_competitor` / `_populate_risk` / `_populate_valuation`；`run_crew` 改同步 + `asyncio.wait_for`；`FLOW_TIMEOUT_SECONDS` 180 → 600；`synthesize_report` 简化为校验 + 填充 runtime 字段 |
| `src/alphaquant/tools/company_lookup_tool.py` | 删 `except Exception` 里的空壳 fallback |
| `src/alphaquant/tools/market_data_tool.py` | 同上 |
| `src/alphaquant/tools/news_tool.py` | 同上 |
| `src/alphaquant/tools/financial_tool.py` | 同上 |
| `src/alphaquant/scoring/__init__.py` | 删 `competitive`, `rating`, `risk_score` 导出 |

### 删除

| 路径 | 原因 |
|---|---|
| `src/alphaquant/scoring/rating.py` | ReportWriter agent 决定 rating/confidence |
| `src/alphaquant/scoring/competitive.py` | CompetitorAnalyst agent 决定 competitive_score |
| `src/alphaquant/scoring/risk_score.py` | RiskAnalyst agent 决定 risk level |
| `src/alphaquant/flows/analysis_flow.py::_populate_competitor` | 不再需要 |
| `src/alphaquant/flows/analysis_flow.py::_populate_risk` | 不再需要 |
| `src/alphaquant/flows/analysis_flow.py::_populate_valuation` | 不再需要 |
| `src/alphaquant/flows/analysis_flow.py::_default_risk_subscores` | 不再需要 |

### 新增

无。Sub-3 主要靠现有模块重组。

### 不动

```
src/alphaquant/core.py
src/alphaquant/main.py
src/alphaquant/interfaces/cli.py
src/alphaquant/interfaces/api/
src/alphaquant/interfaces/frontend/
src/alphaquant/infrastructure/data_sources/    # DataSourceRegistry 不变
src/alphaquant/infrastructure/llm.py            # 仅 timeout 配置
src/alphaquant/infrastructure/config.py         # 仅 litellm_timeout 60 → 120
src/alphaquant/observability/
src/alphaquant/exceptions.py                    # AllDataSourcesDown / ReportGenerationError 不变
src/alphaquant/models/                          # Pydantic schema 不变（除非 ValuationResult.method Literal 需扩展）
src/alphaquant/agents/company_resolver.py       # 4 个 data agent 不变
src/alphaquant/agents/market_analyst.py
src/alphaquant/agents/news_analyst.py
src/alphaquant/agents/financial_analyst.py
src/alphaquant/scoring/dcf.py                   # 保留（ValuationAnalyst 仍调）
src/alphaquant/scoring/financial_health.py     # 保留（ReportWriter 仍调）
```

## 测试计划

### 新增测试

```python
# tests/test_crew.py — 改造
def test_competitor_task_has_output_pydantic(fake_llm):
    """competitor_analyst task 的 output_pydantic 必须是 CompetitorAnalysis。"""

def test_risk_task_has_output_pydantic(fake_llm):
    """risk_analyst task 的 output_pydantic 必须是 RiskAssessment。"""

def test_valuation_task_has_output_pydantic(fake_llm):
    """valuation_analyst task 的 output_pydantic 必须是 ValuationResult。"""

def test_report_writer_task_has_output_pydantic_and_context(fake_llm):
    """report_writer task 的 output_pydantic 必须是 InvestmentReport，且 context 含 task 4/5/6。"""

def test_async_task_indices_cover_data_and_analysis_not_report():
    """_ASYNC_TASK_INDICES == {0,1,2,3,4,5,6}, report writer (idx 7) 串行。"""

# tests/test_flow.py — parse_crew_output 测试改造
def test_parse_crew_output_extracts_competitor_from_pydantic_output():
    """Pydantic output 直接填 state.competitor"""

def test_parse_crew_output_extracts_risk_from_pydantic_output(): ...
def test_parse_crew_output_extracts_valuation_from_pydantic_output(): ...
def test_parse_crew_output_extracts_investment_report_from_pydantic_output(): ...

def test_parse_crew_output_failed_pydantic_sets_none_and_appends_error():
    """LLM 没产出 Pydantic → state.field = None + errors.append"""

# tests/test_flow.py — synthesize_report 测试
def test_synthesize_report_uses_state_report_directly():
    """state.report 已被 ReportWriter 填好，synthesize_report 不重算 rating"""

# tests/test_flow.py — graceful degradation
def test_unknown_ticker_raises_all_data_sources_down():
    """ZZZZZZ 端到端 → AllDataSourcesDown（不是 INTERNAL_ERROR）"""

# tests/test_flow.py — timeout
def test_flow_timeout_seconds_is_600():
    """FLOW_TIMEOUT_SECONDS == 600"""

# tests/test_tools.py — 空壳 fallback 删除
def test_company_lookup_tool_no_empty_shell_fallback():
    """AllDataSourcesDown 不再 fallback 到 {"name": "N/A"}"""
```

### 修改测试

```python
# tests/test_flow.py — TestRunCrewStep
def test_run_crew_uses_sync_kickoff_with_wait_for():
    """run_crew 必须用 asyncio.wait_for + sync kickoff（无 asyncio.to_thread(crew.kickoff)）"""

# tests/test_flow.py — TestParseCrewOutput
# 删：test_extracts_competitor_uses_deterministic_fallback (sub-2 的测试)
# 删：test_extracts_risk_uses_deterministic_fallback
# 删：test_extracts_valuation_uses_deterministic_fallback
# 加：上面 4 个 Pydantic extraction 测试
```

### 保留测试

- `tests/test_scoring.py` 中 dcf / financial_health 的纯函数测试（保留，dcf/financial_health 模块仍存在）
- `tests/test_db.py`, `tests/test_observability.py`, `tests/test_observability_wiring.py`
- `tests/test_api.py`（端到端不变）
- `tests/test_agents.py`（除 4 个 analysis agent 的 backtory 改动）
- `tests/test_crew.py` 大部分（除 `test_tools_mapping` / `test_data_tasks_have_async_execution` 需更新）

### 验证清单

- [ ] `uv run pytest tests/ -q` 通过，新增至少 12 个 test，原 224 不回归
- [ ] `python -m alphaquant AAPL --format json` 输出 `InvestmentReport.rating` 字段是 5 个 Literal 值之一
- [ ] `python -m alphaquant AAPL --format json` 输出 `InvestmentReport.confidence` 是 0-100 整数（**不**总是 82）
- [ ] AAPL/MSFT/TSLA 三个 ticker 的 confidence 数字互不相同（如果 LLM 工作正常）
- [ ] `python -m alphaquant ZZZZZZ` 端到端 raise `AllDataSourcesDown`（不卡 180s 后 INTERNAL_ERROR）
- [ ] `grep DataSourceRegistry src/alphaquant/flows/analysis_flow.py` 返回 0 matches
- [ ] 4 个 analysis agent 的 task output 验证：task_out.pydantic 是正确的 model 实例

## 风险与权衡

1. **MiniMax-M3 Pydantic 稳定性**：LLM 不一定每次都产出严格 Pydantic-valid JSON。Mitigation：依赖 CrewAI 0.203.2 的 `output_pydantic` 内置 retry（通常 1-3 次）。如果还失败，`state.field = None` + error，confidence 会自然变低（LLM 自己评估"我没数据所以 confidence 60"）。

2. **InvestmentReport 字段太多**：markdown min_length=1、catalysts、investment_horizon、disclaimer 都是 LLM 输出。需要调 prompt 才能稳定产出完整 Pydantic。Mitigation：在 plan 阶段用 mock LLM 测试 prompt 调优；可迭代 2-3 轮。

3. **`asyncio.to_thread` 移除风险**：如果 `crew.kickoff` 是 blocking 调用，会阻塞主 event loop → 阻塞 FastAPI 其他请求。Mitigation：MiniMax-M3 API call 本质是 HTTP 阻塞，sub-1 也是这样跑（虽然没有 await），目前没观察到问题。sub-3 仍保留 `asyncio.to_thread(_kickoff_sync)` 包装来让 wait_for 能 cancel。

4. **scoring 模块拆分风险**：`scoring/financial_health.py` 仍被 ReportWriter 用作 tool，需要验证 ReportWriter agent 真的调用。如果不调用，sub-3 后 `InvestmentReport.financial_health_score` 是 LLM 出的。Mitigation：在 agent backtory 里明确说"Use the financial_health_score tool to compute the score"。

5. **3 个 deferred blocker 修复可能产生新问题**：
   - Blocker 1 改同步 kickoff 可能让 FastAPI 不再 non-blocking。Mitigation：保留 `asyncio.to_thread` 包装（见风险 3）。
   - Blocker 3 删空壳 fallback 后，tool 返回 error string，可能让 `parse_crew_output` 误判"成功"为"失败"。Mitigation：sub-2 已有的 error-string detection 已经在 `parse_crew_output._extract_data_field` 里工作正常。

6. **byte-for-byte 一致性破坏**：`InvestmentReport.rating` / `confidence` / `markdown` 不再 byte-for-byte 相同。Sub-1 阶段的约束失效。Mitigation：在 spec 明确说"sub-3 起 LLM 决定这些字段，每次跑都不同"。

## 实施顺序（4 tasks，预计 3-4 天）

1. **Day 1: Task 1 — Pydantic 化 + 简化 parse_crew_output**
   - 改 `_TASK_TEMPLATES` 加 pydantic_model
   - 改 `_ASYNC_TASK_INDICES` 覆盖到 6
   - 改 `_build_tasks` 让 report_writer 有 context
   - 改 `parse_crew_output` 走 Pydantic 路径
   - 删 `_populate_*` 和 `_default_risk_subscores`
   - 改 `synthesize_report` 不算 rating/confidence
   - 改 4 个 analysis agent 的 backtory
   - 改 4 个 data tool 删空壳 fallback
   - 删 3 个 scoring 模块（rating, competitive, risk_score）

2. **Day 2: Task 2 — 修 3 个 deferred blocker**
   - 改 `run_crew` 同步 kickoff
   - 改 `FLOW_TIMEOUT_SECONDS` 180 → 600
   - 删 tool 空壳 fallback
   - 跑测试验证

3. **Day 3: Task 3 — 测试覆盖**
   - 补 Pydantic extraction 测试
   - 改 TestRunCrewStep 验证 sync kickoff
   - 加 graceful degradation E2E 测试（mock ZZZZZZ 路径）
   - 跑完整 test suite，验证 235+ pass

4. **Day 4: Task 4 — 真实 LLM 端到端验证**
   - 跑 AAPL 实跑，确认 LLM 真的产出 Pydantic-valid InvestmentReport
   - 跑 MSFT/TSLA 验证 confidence 不同
   - 跑 ZZZZZZ 验证 graceful degradation
   - 如有 prompt 调优问题，迭代 1-2 轮

## 不在 sub-3 范围（明确划线）

- ❌ 开 `allow_delegation=True` → sub-4
- ❌ 开 CrewAI Memory → sub-4
- ❌ 实现 retry / degrade → sub-4
- ❌ 修改 `InvestmentReport` Pydantic schema 字段（除 `ValuationResult.method` Literal 可能微调）
- ❌ 修改 4 个 data agent
- ❌ 重构 `AnalysisCrew` 之外的 crew 行为
- ❌ 修改 frontend / CLI / API（sub-2 已经用 `parse_crew_output` 的字段，sub-3 仍然兼容）
