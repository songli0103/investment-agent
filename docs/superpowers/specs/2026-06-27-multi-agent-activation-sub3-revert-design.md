# AlphaQuant 多 Agent 激活 — Sub-Project 3 Revert 设计文档

> **范围**：本设计文档是 sub-project 3 的**补遗**，描述为什么放弃 sub-3 原 spec 的"严格 Pydantic 输出"路径、采用了"实用回退"路径、以及最终架构的精确描述。
>
> **引用**：原 spec 在 `docs/superpowers/specs/2026-06-21-multi-agent-activation-sub3-design.md`（本文件不改）。本文件是该 spec 的**实施补遗**——描述"试了什么、为什么改、最终上了什么"。
>
> **时间线**：
> - 2026-06-21：sub-3 原 spec 完成（commit `3f1eaea`）+ 实施 plan 完成（commit `6a0a0fa`）
> - 2026-06-21：原 plan Task 1/2 实施完成（commits `234bea3`、`c5a960d`、`b646b75`、`95fb8ad`、`8c75785`、`dbfac17`、`8d1412e`、`65517dd`）
> - 2026-06-22：real-LLM 测试发现 LLM 在 multi-field 模型上输出结构无效（field 错名 / 对话式文本），CrewAI converter retry-loop 触发 180s timeout → 实用回退（FLOW_TIMEOUT_SECONDS 180→300）
> - 2026-06-27：本补遗 spec 落定，把工作树未提交改动固化为 4 个原子 commit

## 背景

Sub-project 3 原 spec 的目标是：**让 4 个 analysis agent 通过 `Task(output_pydantic=...)` 产出 Pydantic 严格结构化输出；Flow 不再覆盖 agent 输出；ReportWriter agent 直接产出完整 `InvestmentReport`**。

实施过程中发现：

- **MiniMax-M3 在小字段集 Pydantic 模型上稳定**（例如 5 字段的 `ReportWriterOutput`：rating / confidence / horizon / catalysts / markdown）
- **MiniMax-M3 在大字段集 / 嵌套 Pydantic 模型上不稳定**：
  - `CompetitorAnalysis`：5 个字段 + `competitors: list[Competitor]`（嵌套）
  - `RiskAssessment`：6 个 sub-scores（`RiskScore` 列表，每个含 rationale ≥10 字符）
  - `ValuationResult`：含 `assumptions: dict` + DCF/relative 数值
- **不稳定表现**：LLM 输出"看起来像 JSON 但字段名拼错 / 嵌套结构不完整 / 混入对话文本"，CrewAI converter 检测到 Pydantic validate 失败后 retry，最多 3-5 次。**每次 retry 触发一次完整 LLM call**，叠加 7 个 task × 3-5 次 retry → 跑出 180s+ 流程 timeout
- **副作用**：sub-3 Blocker 2（FLOW_TIMEOUT_SECONDS=180 太短）的根因不只是 LLM 慢，更是 converter retry-loop

**结论**：strict Pydantic 在 sub-3 当前 LLM 上不可行。退而求其次：**LLM 驱动 LLM 能稳定产出的字段（5 个 judgment/narrative 字段），Flow 用 deterministic helper 算 LLM 不可靠的 3 个结构化分析字段**。

## 目标

**Goal**：sub-3 收官为可发布状态——
1. **LLM 驱动**：`InvestmentReport` 的 5 个 judgment/narrative 字段（`rating` / `confidence` / `investment_horizon` / `catalysts` / `markdown`）由 ReportWriter agent 通过 `output_pydantic=ReportWriterOutput` 产出
2. **Deterministic 计算**：`InvestmentReport` 的 3 个结构化分析字段（`competitors` / `risk` / `valuation`）由 Flow 用 deterministic helper 从 data fields 算出
3. **Confidence Rubric 生效**：ReportWriter agent 按 `e8efef6` commit 的 5 任务 Confidence Rubric 给出 confidence（含 markdown rationale）
4. **3 个 deferred blocker 全修**：sub-2 留下的 asyncio shutdown race / timeout / empty-shell fallback 全部修复

**Non-goals**（明确不做）：
- ❌ 重试 strict Pydantic on multi-field models（已知失败模式，不重试）
- ❌ 引入新模型 / 改 `InvestmentReport` 已有 schema（除 `ReportWriterOutput` 新增 + `ValuationResult.method` / `CompetitorAnalysis.method` Literal 放宽）
- ❌ 改 4 个 data agent（CompanyResolver / MarketAnalyst / NewsAnalyst / FinancialAnalyst）
- ❌ 改 frontend / CLI / API 契约
- ❌ 改 Confidence Rubric（commit `e8efef6` 已稳定）
- ❌ sub-4 任何内容（`allow_delegation=True` / CrewAI Memory / retry / degrade）

## 已确认的决策

1. **3 个 analysis agent 回退到 text-only**：`competitor_analyst` / `risk_analyst` / `valuation_analyst` task 的 `output_pydantic=None`，产出的 raw text 作为 context 给 ReportWriter task（idx 7）。
2. **ReportWriter task 保留 Pydantic**：但用新 slim 模型 `ReportWriterOutput`（5 字段），不是完整 `InvestmentReport`。
3. **Flow 重新引入 3 个 deterministic helper**（inline in `flows/analysis_flow.py`）：`_compute_competitor_analysis` / `_compute_risk_assessment` / `_compute_valuation`，从 data fields 算 Pydantic 结构化模型。这些 helper **不重新拆成独立模块**——保留在 flow 文件里（commit `b646b75` 删除的 `scoring/{competitive,risk_score}` 模块**不**重新引入；sub-1 时代就是这些逻辑在 flow 顶层）。
4. **`synthesize_report` 简化**：从 5 步骤拼装（compute competitor/risk/valuation → assemble report → fill runtime fields → collect sources → write disclaimer）改为 4 步骤（compute 3 analysis → assemble report from data + analyses + writer_output → fill runtime fields）。
5. **`_ASYNC_TASK_INDICES` 调整**：从 `{0,1,2,3}` → `{0,1,2,3,4,5,6}`（保留：3 个 analysis text-only 任务并行加速，但 `parse_crew_output` 不再读它们的输出）。
6. **`FLOW_TIMEOUT_SECONDS` 180 → 300**：实测 7 个 LLM task（含 1 个 sequential report_writer）300s 内可完成；600s 不必要（300 留 2-3× 缓冲）。
7. **3 个 deferred blocker 修复策略**：
   - **Blocker 1（asyncio shutdown race）**：`run_crew` 把 `crew.kickoff(inputs=...)` 包在 sync helper `_kickoff_sync()` + `asyncio.to_thread()`，外层 `asyncio.wait_for(timeout)` 可 cancel mid-execution。`parse_crew_output` 保持纯同步（已经是了，只确保无隐式 await）。
   - **Blocker 2（timeout）**：见决策 6（300s）。
   - **Blocker 3（tool empty-shell fallback）**：4 个 data tool 的 `except Exception` 分支删掉空壳 dict fallback（`{"name": "N/A", ...}`），改返回 error string `"Error fetching X: {type}: {e}"`。`parse_crew_output._extract_data_field` 已有 error-string 检测（sub-2 既有逻辑保留）。
8. **`_coerce_rating` validator 新增**：LLM 输出不在 `{Strong Buy, Buy, Hold, Sell, Strong Sell}` 时兜底为 `"Hold"`（不抛 ValidationError）。参考 `CompetitorAnalysis._coerce_method` 已建立的 pattern。
9. **`confidence: int | None` 允许 None**：ReportWriter agent 可输出 `null`（"我对这个 confidence 没把握"），`InvestmentReport.confidence` 透传。前端 / DB 已支持 NULL（commit `d2e2ab8` 迁移）。

## 架构

### 数据流图

```
run_crew (@start):
  _kickoff_sync() (在 worker thread):
    AnalysisCrew().kickoff(inputs={"ticker": normalized})
      ├─ async tasks {0..6}:
      │   ├─ 0: company_resolver  → state.company
      │   ├─ 1: market_analyst    → state.market
      │   ├─ 2: news_analyst      → state.news
      │   ├─ 3: financial_analyst → state.financial
      │   ├─ 4: competitor_analyst (text-only, IGNORED in parse_crew_output)
      │   ├─ 5: risk_analyst       (text-only, IGNORED in parse_crew_output)
      │   └─ 6: valuation_analyst  (text-only, IGNORED in parse_crew_output)
      └─ sequential task 7:
          └─ report_writer
              context=[task 4, task 5, task 6]
              output_pydantic=ReportWriterOutput
              → state.writer_output
  parse_crew_output (sync):
    tasks[0..3] → state.{company, market, news, financial}
    tasks[4..6] → IGNORED (text used as context for report_writer)
    tasks[7]    → state.writer_output

synthesize_report (@listen(run_crew)):
  state.competitor = _compute_competitor_analysis(state)   # deterministic
  state.risk       = _compute_risk_assessment(state)       # deterministic
  state.valuation  = _compute_valuation(state)            # deterministic
  state.report     = InvestmentReport(...)                # data + 3 analyses + writer_output (inline)
  state.report.disclaimer = DISCLAIMER_TEXT
  state.report.generated_at = datetime.utcnow()
  state.report.report_id = str(uuid.uuid4())
  state.report.sources = _collect_sources(state.market, state.news, state.financial, state.competitor)
```

### Task 模板最终形态

```python
# crews/analysis_crew.py
_TASK_TEMPLATES: list[tuple[str, str, type[BaseModel] | None]] = [
    # (role_key, description_template, output_pydantic_model_or_None)
    ("company_resolver",  "Validate ticker '{ticker}' and return canonical company metadata.", None),
    ("market_analyst",    "Fetch market data for '{ticker}'.", None),
    ("news_analyst",      "Fetch recent news for '{ticker}'.", None),
    ("financial_analyst", "Fetch financial statements for '{ticker}'.", None),
    # 3 analysis tasks: text-only (LLM 不可靠产出严格 Pydantic，text 作为 context 给 report_writer)
    ("competitor_analyst","Summarize the competitive landscape for '{ticker}' in plain text. ...", None),
    ("risk_analyst",      "Summarize the key risk factors for '{ticker}' in plain text. ...",       None),
    ("valuation_analyst", "Summarize the valuation analysis (DCF + relative) for '{ticker}' ...",  None),
    # ReportWriter: Pydantic slim model (5 字段，LLM 能稳定产出)
    ("report_writer",     "Synthesize ReportWriterOutput for '{ticker}'.", ReportWriterOutput),
]

_REPORT_WRITER_INDEX = 7
_ASYNC_TASK_INDICES = {0, 1, 2, 3, 4, 5, 6}  # 含 3 个 analysis text-only 任务并行

# report_writer task 构造时传 context
Task(
    description=...,
    expected_output="...",  # 仍写字符串，但 Pydantic 实际产出
    agent=report_writer_agent,
    context=[self.tasks[i] for i in (4, 5, 6)],
    async_execution=False,
)
```

### Agent ↔ Tool 映射

| Agent | Tools | 变化 |
|---|---|---|
| CompanyResolver | `[CompanyLookupTool()]` | 不变 |
| MarketAnalyst | `[MarketDataTool()]` | 不变 |
| NewsAnalyst | `[NewsTool()]` | 不变 |
| FinancialAnalyst | `[FinancialTool()]` | 不变 |
| CompetitorAnalyst | `[CompetitorTool()]` | backtory 改：强调"text-only, Flow 算结构化" |
| RiskAnalyst | `[]` | backtory 改：同上 |
| ValuationAnalyst | `[DCFTool()]` | backtory 改：同上 |
| ReportWriter | `[]` | backtory 改：强调"output_pydantic=ReportWriterOutput 5 字段" |

### Pydantic schema

**新增**（`src/alphaquant/models/report.py`）：

```python
class ReportWriterOutput(BaseModel):
    """LLM-produced subset of InvestmentReport. Sub-3 revert: the structured
    analysis fields (competitors, risk, valuation) are computed deterministically
    by the flow; the LLM only produces the synthesis fields below."""

    rating: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    confidence: int | None = Field(None, ge=0, le=100)
    investment_horizon: Literal["short", "medium", "long"] = "medium"
    catalysts: list[str] = Field(default_factory=list)
    markdown: str = Field(..., min_length=1)

    @field_validator("rating", mode="before")
    @classmethod
    def _coerce_rating(cls, v: Any) -> Any:
        allowed = {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}
        return v if v in allowed else "Hold"
```

**已有但放宽**（`src/alphaquant/models/{valuation,competitor}.py`）：
- `ValuationResult.method` Literal 放宽到 `{dcf_relative_peg, relative_only, blended, dcf_only, relative, dcf_relative_blended}`（commit `dbfac17`）；越界值由 `_coerce_method` 兜底成 `"dcf_relative_peg"`
- `CompetitorAnalysis.method` Literal 放宽到 `{gics, keyword, manual, fallback, hybrid, multi_factor, peer_comparison}`（commit `dbfac17`）；越界值由 `_coerce_method` 兜底成 `"gics"`
- `RiskAssessment` 6 个 sub-score 约束放宽（commit `65517dd`）

**不变**：
- `InvestmentReport` 字段（除 `confidence: int | None` 已允许 None，commit `de51986`）
- 4 个 data 模型（`Company` / `MarketData` / `FinancialStatements` / `NewsAnalysis`）

### Deterministic helper 签名（Flow 内部 inline）

```python
# flows/analysis_flow.py（inline，非独立模块）

def _gics_peers_for(ticker: str, sector: str | None) -> list[Competitor]:
    """Fallback peer set（当 market.peers 为空时用）— 来自 sub-1 GICS_PEERS map"""

def _compute_competitor_analysis(state: "AnalysisState") -> CompetitorAnalysis | None:
    """从 state.market.peers + state.company.industry 算 competitive_score。
    失败或缺数据 → None + state.errors.append("competitor_unavailable")"""

def _default_risk_subscores(state: "AnalysisState") -> list[RiskScore]:
    """6 类 RiskScore 兜底（financial / operational / market / regulatory / governance / macro）"""

def _compute_risk_assessment(state: "AnalysisState") -> RiskAssessment | None:
    """_default_risk_subscores + weighted_sum → RiskAssessment.total_score"""

def _compute_valuation(state: "AnalysisState") -> ValuationResult | None:
    """compute_dcf_value(fcf, growth, shares, wacc, g_term) + relative value。
    method = "dcf_relative_peg" if dcf_value else "relative_only" """

def _assemble_report(state: "AnalysisState") -> InvestmentReport:
    """从 state.{company,market,news,financial,competitor,risk,valuation,writer_output} 拼装。
    compute_financial_health(state.financial) → state.report.financial_health_score"""

# 注:实际工作树里 `_assemble_report` 是 inline 在 `synthesize_report` 里,
# 不抽成独立函数。spec 保留这个签名是"应该的样子",实际代码可能 inline。
```

## 数据流细节

### `parse_crew_output` 最终形态

```python
def parse_crew_output(result: CrewOutput, state: "AnalysisState") -> dict[str, Any]:
    """Sub-3 revert: 4 data fields from agent task outputs + writer_output only.
    3 analysis tasks' text output IGNORED (used as context for report_writer)."""
    tasks_output = result.tasks_output or []
    extracted: dict[str, Any] = {}

    # 4 data fields (sub-2 unchanged)
    _extract_data_field(tasks_output, 0, "company_resolver",  Company,             state, raise_on_fail=True)
    _extract_data_field(tasks_output, 1, "market_analyst",    MarketData,          state)
    _extract_news_field(tasks_output, 2, "news_analyst",      state)
    _extract_data_field(tasks_output, 3, "financial_analyst", FinancialStatements, state)

    # ReportWriter output (sub-3 revert: ReportWriterOutput, not full InvestmentReport)
    state.writer_output = _extract_pydantic_field(
        tasks_output, 7, "report_writer", ReportWriterOutput, state
    )

    return extracted
```

### `synthesize_report` 最终形态

```python
@listen(run_crew)
def synthesize_report(self) -> None:
    """Sub-3 revert: compute 3 analysis fields deterministically, then assemble
    InvestmentReport from data + analyses + writer_output."""
    # Step 1: deterministic analysis fields (inline helpers)
    self.state.competitor = _compute_competitor_analysis(self.state)
    self.state.risk       = _compute_risk_assessment(self.state)
    self.state.valuation  = _compute_valuation(self.state)

    # Step 2: assemble full report inline (raises ReportGenerationError if writer_output is None)
    self.state.report = InvestmentReport(
        company=..., market=..., ...,
        rating=self.state.writer_output.rating,
        confidence=self.state.writer_output.confidence,
        ...
    )

    # Step 3: fill runtime fields
    self.state.report.disclaimer = DISCLAIMER_TEXT
    self.state.report.generated_at = datetime.utcnow()
    self.state.report.report_id = str(uuid.uuid4())
    self.state.report.sources = _collect_sources(
        self.state.market, self.state.news, self.state.financial, self.state.competitor
    )
```

## 修复 3 个 Deferred Blocker

### Blocker 1: `parse_crew_output` asyncio shutdown race

**修复**（已存在工作树中，commit 3 实现）：

```python
@start()
async def run_crew(self) -> None:
    # ... normalize ticker ...

    def _kickoff_sync() -> CrewOutput:
        # 把 crew.kickoff 的实际调用挪进 sync helper
        return AnalysisCrew().kickoff(inputs={"ticker": normalized})

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_kickoff_sync),
            timeout=FLOW_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.error("crew_timeout", ticker=normalized)
        raise

    parse_crew_output(result, self.state)  # 纯同步
```

### Blocker 2: `FLOW_TIMEOUT_SECONDS` 180 太短

**修复**（已存在工作树中）：180 → 300。
- 7 个 LLM task × 平均 20-40s = 140-280s
- 加 CrewAI 内部开销（manager decision / converter retry）= 50-100s
- 300s 留 1.5-2× 缓冲（vs 原 spec 600s 留 3-4× 缓冲太保守）

### Blocker 3: tool 返回空壳而非 `AllDataSourcesDown`

**修复**（commit `8d1412e`，已 committed）：tool 在 `except Exception` 分支返回 error string 而非 `{"name": "N/A", ...}` 空壳 dict。`parse_crew_output._extract_data_field` 已有 error-string detection 保留。

## 失败处理

| 失败模式 | 处理 |
|---|---|
| Tool 抛 `AllDataSourcesDown` | 转 error string → `parse_crew_output` 检测后 raise `AllDataSourcesDown`（sub-2 既有逻辑） |
| Tool 30s timeout | 转 error string → 同上 |
| Tool 抛其他 Exception | error string（不再是空壳 dict） |
| 4 个 data agent 失败 | `state.field = None` + `errors.append("xxx_unavailable")` |
| 3 个 analysis agent 失败 | **不再读它们输出**（text-only）→ 视为"软失败"，不影响 Flow 算 competitor/risk/valuation |
| ReportWriter agent 失败（Pydantic 解析失败） | `_extract_pydantic_field` 检测后 `state.writer_output = None` + `errors.append("writer_output_unavailable")` |
| `_assemble_report` 时 `writer_output is None` | raise `ReportGenerationError`（FastAPI 转 500 / CLI 转 INTERNAL_ERROR） |
| Crew hang | `asyncio.wait_for(300s)` 触发 `asyncio.TimeoutError` → 同 `ReportGenerationError` |
| LLM 输出 rating 不在 Literal | `_coerce_rating` 兜底为 `"Hold"`（不抛 ValidationError） |

## vs 原 sub-3 spec 的偏离总表

| 项 | 原 spec | 最终实现 | 理由 |
|---|---|---|---|
| Task 4/5/6 output | Pydantic(strict) | text-only + context | LLM 输出结构无效触发 retry-loop 到 timeout |
| Task 7 output | `InvestmentReport`(full) | `ReportWriterOutput`(slim 5 字段) | LLM 在大字段集上不可靠；瘦模型 + 兜底 validator 可稳定 |
| `parse_crew_output` 对 competitor/risk/valuation | Pydantic extract | IGNORED | 同上 |
| `synthesize_report` | 仅填 runtime 字段 | 算 3 analysis + 拼装 | LLM 不再产出 competitor/risk/valuation |
| `FLOW_TIMEOUT_SECONDS` | 180 → 600 | 180 → 300 | 实测 300 够；600 保守 |
| `scoring/{rating,competitive,risk_score}` | 删除 | **保留删除**；deterministic 计算以 inline `_compute_*` helper 形式存在于 `flows/analysis_flow.py` | 复用 sub-1 时代的 flow-internal 实现位置（最内聚） |
| `_coerce_rating` validator | 无 | 新增 | LLM 输出越界 rating 时兜底 "Hold" |
| `_ASYNC_TASK_INDICES` | `{0..3}` → `{0..6}` | `{0..3}` → `{0..6}`（**不变**） | text-only 任务并行仍加速（无 Pydantic converter 开销） |
| `confidence: int \| None` 允许 None | 已在 commit `de51986` 支持 | 不变 | Confidence Rubric 兼容 |

## 文件变更清单

### 修改（commit 1-3）

| 路径 | 变更 | 对应 commit |
|---|---|---|
| `src/alphaquant/models/report.py` | 新增 `ReportWriterOutput` 类 + `_coerce_rating` validator | commit 1 |
| `src/alphaquant/crews/analysis_crew.py` | `_TASK_TEMPLATES` 调整（3 个 analysis text-only + report_writer Pydantic=ReportWriterOutput）；`_ASYNC_TASK_INDICES = {0..6}`；report_writer task 加 `context=[tasks[4..6]]` | commit 2 |
| `src/alphaquant/agents/competitor_analyst.py` | backtory 改：text-only | commit 2 |
| `src/alphaquant/agents/risk_analyst.py` | backtory 改：text-only | commit 2 |
| `src/alphaquant/agents/valuation_analyst.py` | backtory 改：text-only | commit 2 |
| `src/alphaquant/agents/report_writer.py` | backtory 改：output_pydantic=ReportWriterOutput 5 字段 | commit 2 |
| `src/alphaquant/flows/analysis_flow.py` | 重新引入 inline `_compute_competitor_analysis` / `_compute_risk_assessment` / `_compute_valuation` / `_assemble_report`；`FLOW_TIMEOUT_SECONDS = 300`；`run_crew` 改用 `_kickoff_sync` + `asyncio.to_thread` + `asyncio.wait_for`；`parse_crew_output` 只读 4 data + 1 writer_output；`synthesize_report` 4 步骤 | commit 3 |
| `src/alphaquant/models/valuation.py` | `method` Literal 放宽到 `{dcf_relative_peg, relative_only, blended, dcf_only, relative, dcf_relative_blended}`（commit `dbfac17` 已在 main） | commit 3 |
| `src/alphaquant/models/competitor.py` | `method` Literal 放宽到 `{gics, keyword, manual, fallback, hybrid, multi_factor, peer_comparison}`（commit `dbfac17` 已在 main） | commit 3 |
| `src/alphaquant/models/__init__.py` | export `ReportWriterOutput` | commit 3 |
| `src/alphaquant/interfaces/frontend/pages/1_Analyze.py` | 适配新模型字段（看实际 diff） | commit 3 |

### 删除

**无**。本 spec 不删除文件；deterministic helpers 作为 inline 函数加在 `flows/analysis_flow.py`，不重新引入独立 `scoring/{competitive,risk_score}` 模块。

### 测试变更（commit 4）

| 路径 | 变更 |
|---|---|
| `tests/test_flow.py` | 改 TestParseCrewOutput 测试（5 字段 Pydantic 提取而非 4 个 analysis Pydantic）；新增 TestAssembleReport 测试；更新 timeout 测试到 300 |
| `tests/test_crew.py` | 改 test_task_templates（task 4-6 Pydantic=None，task 7=ReportWriterOutput）；改 test_async_indices |
| `tests/test_models_literals.py` | 更新放宽的 Literal 断言 |
| `tests/conftest.py` | 抽 InvestmentReport fixture（如尚未抽，看 commit `eb77cdc` 状态） |

### 不动

```
src/alphaquant/core.py
src/alphaquant/main.py
src/alphaquant/interfaces/cli.py
src/alphaquant/interfaces/api/
src/alphaquant/infrastructure/data_sources/    # DataSourceRegistry 不变
src/alphaquant/infrastructure/llm.py            # 仅 timeout 配置（在 commit 1/2 不动）
src/alphaquant/observability/                   # 不动
src/alphaquant/exceptions.py                    # AllDataSourcesDown / ReportGenerationError 不变
src/alphaquant/models/company.py                # 4 个 data 模型不动
src/alphaquant/models/financial.py
src/alphaquant/models/market.py
src/alphaquant/models/news.py
src/alphaquant/models/risk.py                   # RiskAssessment 约束已在 commit 65517dd 放宽
src/alphaquant/agents/company_resolver.py       # 4 个 data agent 不动
src/alphaquant/agents/market_analyst.py
src/alphaquant/agents/news_analyst.py
src/alphaquant/agents/financial_analyst.py
src/alphaquant/scoring/dcf.py                   # 保留（Flow _populate_valuation 调用）
src/alphaquant/scoring/financial_health.py     # 保留（Flow _assemble_report 调用）
src/alphaquant/frontend/                        # 仅 1_Analyze.py 微调（commit 3）
```

## 严格输出一致性

`InvestmentReport` 非时间戳/UUID 字段不再 byte-for-byte 相同 — sub-3 起 LLM 驱动 5 个 judgment/narrative 字段，每次跑都不同（这是 LLM 决定的本质，不是 bug）。

**保留** byte-for-byte 一致性的字段：
- `company.*` / `market.*` / `news.*` / `financial.*`（仍走 4 个 data agent 抓数据）
- `competitors.competitors` 列表（由 `_populate_competitor` 从 `market.peers` 算）
- `risk.sub_scores`（由 `_populate_risk` 算）
- `valuation.{dcf_value, intrinsic_value_per_share, current_price, upside_pct, relative_value, peg_ratio}`（由 `_populate_valuation` 算）

**LLM-决定** 字段（每次跑都不同）：
- `rating`, `confidence`, `investment_horizon`, `catalysts`, `markdown`, `sources`（汇总）, `disclaimer`（写死中文）

## 验证清单

实施完成后，下列验证必须全部通过：

### 测试 suite

- [ ] `uv run pytest tests/ -q --tb=short` 通过，≥250 passed, 0 failed
- [ ] `tests/test_flow.py::TestGracefulDegradation::test_unknown_ticker_raises_all_data_sources_down` 通过（ZZZZZZ → AllDataSourcesDown）
- [ ] `tests/test_observability.py::test_company_failure_logs_all_data_sources_down_event` 通过（已 commit `5553e43`）

### CLI / API 端到端（真实 LLM）

- [ ] `python -m alphaquant AAPL --format json` 输出包含 `rating` ∈ {Strong Buy, Buy, Hold, Sell, Strong Sell}、`confidence` 是 0-100 整数或 null、`markdown` 非空字符串
- [ ] `python -m alphaquant MSFT --format json` 同上
- [ ] `python -m alphaquant TSLA --format json` 同上
- [ ] 3 个 ticker 的 `confidence` 数字**互不相同**（如果 LLM 工作正常）
- [ ] 3 个 ticker 的 `rating` / `markdown` / `catalysts` 内容不同（LLM 决定）
- [ ] `python -m alphaquant ZZZZZZ` 端到端 raise `AllDataSourcesDown`（<30s 失败，不卡 300s 后 INTERNAL_ERROR）
- [ ] AAPL 跑耗时 <300s

### 代码搜索

- [ ] `grep -r "scoring.rating" src/alphaquant/flows/` 返回 0 matches（rating 由 LLM 决定）
- [ ] `grep -r "deterministic_fallback" src/alphaquant/flows/` 返回 0 matches（旧术语已废弃）
- [ ] `grep "FLOW_TIMEOUT_SECONDS = 300" src/alphaquant/flows/analysis_flow.py` 1 match
- [ ] `grep "_coerce_rating" src/alphaquant/models/report.py` 1 match

## 风险与权衡

1. **LLM 输出 markdown 长度 / 风格不一致**：每次跑 markdown 不同，frontend 缓存逻辑需要确认兼容。Mitigation：commit 4 测试覆盖 markdown 非空约束。

2. **deterministic helpers 作为 `_compute_*` 函数 inline 在 flow**：和 sub-1 时代位置相同（sub-1 这些 helper 就在 flow 文件里；sub-3 试图拆到 `scoring/{competitive,risk_score}` 后删除；sub-3 revert 回到 sub-1 的位置）。Mitigation：本 spec 明确说明这个偏离，避免后续维护者困惑"为什么 sub-3 拆了又合回来"。

3. **3 个 analysis agent 产出的 text 现在**没用**：它们的输出仅作为 report_writer 的 context（`context=[tasks[4..6]]`）。如果 LLM 在 task 4-6 上 hang / 慢，整个 run 仍会卡。Mitigation：3 个 analysis task 仍 `async_execution=True`，并行启动；如果 LLM hang 会被 `FLOW_TIMEOUT_SECONDS=300` 兜住。

4. **`FLOW_TIMEOUT_SECONDS=300` 对未来更慢的 LLM 仍可能不够**：Mitigation：可配常量 `infrastructure/config.py`，后续改不动 flow。本 spec 暂不动这个配置（不在范围）。

5. **`_coerce_rating` 兜底掩盖 LLM 错误**：如果 LLM 长期输出无效 rating，会被 silently 改成 "Hold"，下游看不到错误。Mitigation：未来加 observability 事件（不在本 spec 范围）。

6. **byte-for-byte 一致性破坏**：`InvestmentReport.rating` / `confidence` / `markdown` 不再 byte-for-byte 相同。这是 LLM 决定字段的本质，sub-1 时代的约束失效。Mitigation：本 spec §"严格输出一致性"明确说明。

## 不在 sub-3-revert 范围（明确划线）

- ❌ 重试 strict Pydantic on multi-field models
- ❌ 引入新模型 / 改 `InvestmentReport` 已有 schema（除 `ReportWriterOutput` 新增 + `ValuationResult.method` / `CompetitorAnalysis.method` Literal 放宽）
- ❌ 改 4 个 data agent（CompanyResolver / MarketAnalyst / NewsAnalyst / FinancialAnalyst）
- ❌ 改 frontend / CLI / API 契约
- ❌ 改 Confidence Rubric（commit `e8efef6`）
- ❌ sub-4 任何内容（`allow_delegation=True` / CrewAI Memory / retry / degrade）

## 子项目 4 范围预告（独立 spec）

- `allow_delegation=True` 让 manager agent 重新分发任务
- CrewAI Memory（短期 + 长期 + entity memory）
- Retry strategy（LLM 输出无效时多试几次，带 backoff）
- 渐进 degrade（部分 tool 失败时仍出报告，confidence 自然降低）

## 实施顺序（4 commits + 1 验证）

### Commit 1: `feat(models): add ReportWriterOutput slim Pydantic for sub-3 revert`
- `src/alphaquant/models/report.py`：新增 `ReportWriterOutput` 类 + `_coerce_rating` validator
- 验证：`uv run pytest tests/test_models_literals.py -q`

### Commit 2: `feat(crew): revert 3 analysis tasks to text-only + slim report_writer Pydantic`
- `src/alphaquant/crews/analysis_crew.py`：`_TASK_TEMPLATES` 调整、`_ASYNC_TASK_INDICES = {0..6}`、report_writer context
- `src/alphaquant/agents/{competitor,risk,valuation,report_writer}.py`：backtory 改
- 验证：`uv run pytest tests/test_crew.py -q`

### Commit 3: `feat(flow): compute competitor/risk/valuation deterministically + assemble report from writer_output`
- `src/alphaquant/flows/analysis_flow.py`：inline `_compute_competitor_analysis` / `_compute_risk_assessment` / `_compute_valuation` + inline `synthesize_report` 拼装 `InvestmentReport` + 改 `run_crew` 用 `_kickoff_sync` + `asyncio.to_thread` + `asyncio.wait_for` + `FLOW_TIMEOUT_SECONDS = 300`
- `src/alphaquant/models/{valuation,competitor}.py`：Literal 放宽（已在 main，不需 commit 3 改）
- `src/alphaquant/models/__init__.py`：export `ReportWriterOutput`
- `src/alphaquant/interfaces/frontend/pages/1_Analyze.py`：适配
- 验证：`uv run pytest tests/test_flow.py -q`

### Commit 4: `test(sub-3): update fixtures + trailing newline + literal widening`
- `tests/test_flow.py` / `tests/test_crew.py` / `tests/test_models_literals.py` / `tests/conftest.py`（如需要）
- 验证：`uv run pytest tests/ -q`

### Validation: 3 real tickers + 1 failure
- `python -m alphaquant AAPL --format json`
- `python -m alphaquant MSFT --format json`
- `python -m alphaquant TSLA --format json`
- `python -m alphaquant ZZZZZZ`（期望 AllDataSourcesDown <30s）
- 验证 §"验证清单"

## 引用

- 原 sub-3 spec：`docs/superpowers/specs/2026-06-21-multi-agent-activation-sub3-design.md`
- 原 sub-3 plan：`docs/superpowers/plans/2026-06-21-multi-agent-activation-sub3.md`
- sub-2 spec：`docs/superpowers/specs/2026-06-21-multi-agent-activation-sub2-design.md`
- sub-2 plan：`docs/superpowers/plans/2026-06-21-multi-agent-activation-sub2.md`
- Confidence Rubric spec：`docs/superpowers/specs/2026-06-21-confidence-rubric-design.md`
- Confidence Rubric plan：`docs/superpowers/plans/2026-06-21-confidence-rubric.md`
