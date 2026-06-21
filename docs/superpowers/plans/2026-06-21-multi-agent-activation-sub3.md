# Multi-Agent Activation — Sub-Project 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 4 analysis agents (CompetitorAnalyst, RiskAnalyst, ValuationAnalyst, ReportWriter) produce real LLM reasoning output via `Task(output_pydantic=...)`; remove all deterministic Flow-side fallbacks; ReportWriter agent decides all `InvestmentReport` fields (rating, confidence, catalysts, investment_horizon, markdown). Also fix the 3 deferred blockers from sub-2 (asyncio shutdown race, FLOW_TIMEOUT_SECONDS too short, tool empty-shell fallback).

**Architecture:** Extend `_TASK_TEMPLATES` with `output_pydantic` model column; expand `_ASYNC_TASK_INDICES` to cover data + analysis tasks (0-6); `ReportWriter` (idx 7) runs sequentially with `context=[tasks[4..6]]`; `parse_crew_output` introduces `_extract_pydantic_field` helper that pulls the Pydantic instance from `task_out.pydantic` only (NO raw-JSON fallback per spec §"完全删除 fallback"); `_populate_competitor` / `_populate_risk` / `_populate_valuation` / `_default_risk_subscores` deleted; `synthesize_report` becomes a thin runtime-field-fill step. Blocker 1 fix: wrap `crew.kickoff(inputs=...)` in `_kickoff_sync()` + `asyncio.to_thread()` so `asyncio.wait_for` can cancel mid-execution. Blocker 2: `FLOW_TIMEOUT_SECONDS` 180 → 600. Blocker 3: drop `except Exception` empty-shell fallback in 4 data tools — they return error string instead. Test strategy: mock tools + `_FakeLLM` for unit tests; smoke real LLM for end-to-end validation.

**Decision recorded 2026-06-21 during pre-flight:** Strict no-fallback. If `task_out.pydantic` is `None` or wrong type, append `"<key>_unavailable"` to `state.errors` and return `None` — never try to recover by parsing `task_out.raw`. If LLM cannot produce schema-valid JSON, that's an LLM problem, not a flow problem.

**Tech Stack:** Python 3.11, CrewAI 0.203.2 (installed), LiteLLM, Pydantic v2, asyncio.

## Global Constraints

- 4 analysis agents use `Task(output_pydantic=...)` — no LLM-side summarization of tool JSON (that's data agents in sub-2)
- No `allow_delegation=True`, no CrewAI Memory (those are sub-4)
- No retry / degrade logic (sub-4)
- Do NOT modify: `core.py`, `interfaces/cli.py`, `interfaces/api/`, `interfaces/frontend/`, `infrastructure/data_sources/`, `infrastructure/llm.py` (only `infrastructure/config.py` may update `litellm_timeout`), `main.py`, `exceptions.py`, `models/company.py`, `models/financial.py`, `models/market.py`, `models/news.py`
- Do NOT modify 4 data agents (CompanyResolver, MarketAnalyst, NewsAnalyst, FinancialAnalyst) — sub-2 already wired them
- `InvestmentReport` Pydantic schema unchanged, but `ValuationResult.method` Literal may be widened in Task 1 if LLM produces values outside `["dcf_relative_peg", "relative_only"]` (this is allowed schema extension, not a breaking change)
- 3 deferred blockers from sub-2 must be fixed: async race in `parse_crew_output`, `FLOW_TIMEOUT_SECONDS=180` too short, data tool empty-shell fallback
- `FLOW_TIMEOUT_SECONDS`: 180 → 600
- `_ASYNC_TASK_INDICES`: `{0,1,2,3}` → `{0,1,2,3,4,5,6}` (data + analysis parallel; report writer serial)
- `ReportWriter` (task idx 7) `context=[self.tasks[4], self.tasks[5], self.tasks[6]]` so it receives Competitor/Risk/Valuation Pydantic outputs as upstream context
- Test baseline: 224 passing (must not regress); target ≥ 236 after sub-3
- LLM failures → `state.<field> = None` + `state.errors.append("<key>_unavailable")`; ReportWriter failure → `synthesize_report` raises `ReportGenerationError`
- Tool `_run` keeps `AllDataSourcesDown` handling (already in sub-2); only the `except Exception` empty-shell fallback is removed in Task 3

---

### Task 1: Crew Pydantic schema + 4 analysis agent backtory rewrite

**Files:**
- Modify: `src/alphaquant/crews/analysis_crew.py:32-122` (full rewrite of `_TASK_TEMPLATES` and `_build_tasks`)
- Modify: `src/alphaquant/agents/competitor_analyst.py:17-20` (backtory only)
- Modify: `src/alphaquant/agents/risk_analyst.py:14-18` (backtory only)
- Modify: `src/alphaquant/agents/valuation_analyst.py:17-21` (backtory only)
- Modify: `src/alphaquant/agents/report_writer.py:16-19` (backtory only)
- Modify: `src/alphaquant/models/valuation.py` (only if LLM produces `method` values outside current Literal — see Step 9)
- Modify: `tests/test_crew.py` (add 5 tests for output_pydantic + async indices + context wiring)
- Test: `tests/test_crew.py`

**Interfaces:**
- Consumes: existing `build_*_agent(llm)` signatures in `src/alphaquant/agents/*.py`
- Produces:
  - `AnalysisCrew._TASK_TEMPLATES: list[tuple[str, str, type[BaseModel] | None]]` — 3-tuple (role_key, description, pydantic_model_or_None)
  - `AnalysisCrew._ASYNC_TASK_INDICES: set[int] = {0, 1, 2, 3, 4, 5, 6}` (as class-level constant; addresses sub-2 Minor M1)
  - `AnalysisCrew.tasks[7].context = [tasks[4], tasks[5], tasks[6]]` (ReportWriter depends on 3 analysis tasks)

---

#### Step 1: Write failing test for `_TASK_TEMPLATES` 3-tuple shape and `_ASYNC_TASK_INDICES`

In `tests/test_crew.py`, find the existing `class TestAnalysisCrew:` (or similar) and **add** the following test method at the end of the class. Do NOT remove existing tests. If existing tests reference the old 2-tuple `_TASK_TEMPLATES`, leave them — they exercise the runtime task list (still 8 items), not the source constant.

```python
def test_task_templates_uses_3_tuple_with_pydantic_model(self):
    """_TASK_TEMPLATES entries must be (key, description, pydantic_model_or_None)."""
    from alphaquant.crews.analysis_crew import _TASK_TEMPLATES
    from alphaquant.models.competitor import CompetitorAnalysis
    from alphaquant.models.risk import RiskAssessment
    from alphaquant.models.valuation import ValuationResult
    from alphaquant.models.report import InvestmentReport

    assert len(_TASK_TEMPLATES) == 8
    for entry in _TASK_TEMPLATES:
        assert len(entry) == 3, f"expected 3-tuple, got {len(entry)}-tuple: {entry!r}"

    keys = [t[0] for t in _TASK_TEMPLATES]
    assert keys == [
        "company_resolver", "market_analyst", "news_analyst", "financial_analyst",
        "competitor_analyst", "risk_analyst", "valuation_analyst", "report_writer",
    ]

    # 4 data tasks: no Pydantic output (tool JSON only)
    assert _TASK_TEMPLATES[0][2] is None
    assert _TASK_TEMPLATES[1][2] is None
    assert _TASK_TEMPLATES[2][2] is None
    assert _TASK_TEMPLATES[3][2] is None
    # 3 analysis tasks + report writer: Pydantic
    assert _TASK_TEMPLATES[4][2] is CompetitorAnalysis
    assert _TASK_TEMPLATES[5][2] is RiskAssessment
    assert _TASK_TEMPLATES[6][2] is ValuationResult
    assert _TASK_TEMPLATES[7][2] is InvestmentReport


def test_async_task_indices_cover_data_and_analysis_not_report(self):
    """_ASYNC_TASK_INDICES must cover 0-6 (data + analysis), not 7 (report writer)."""
    from alphaquant.crews.analysis_crew import AnalysisCrew

    # Build crew with a fake LLM to avoid network calls
    from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
    # Patch get_llm to return a deterministic stub
    from unittest.mock import patch
    from alphaquant.crews.analysis_crew import get_llm as _real_get_llm
    from tests.conftest import _FakeLLM
    fake = _FakeLLM()
    with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
        crew = _AC()
    async_indices = getattr(crew, "_ASYNC_TASK_INDICES", None)
    assert async_indices is not None, "_ASYNC_TASK_INDICES must be a class-level constant"
    assert async_indices == {0, 1, 2, 3, 4, 5, 6}
    assert 7 not in async_indices  # report writer is serial
```

#### Step 2: Run tests to verify they fail

Run: `uv run pytest tests/test_crew.py::TestAnalysisCrew::test_task_templates_uses_3_tuple_with_pydantic_model tests/test_crew.py::TestAnalysisCrew::test_async_task_indices_cover_data_and_analysis_not_report -v`
Expected: FAIL — current `_TASK_TEMPLATES` is a 2-tuple list and `_ASYNC_TASK_INDICES` is a local variable inside `_build_tasks`.

#### Step 3: Rewrite `_TASK_TEMPLATES` to 3-tuple and lift `_ASYNC_TASK_INDICES` to class constant

In `src/alphaquant/crews/analysis_crew.py`, **replace** the entire `_TASK_TEMPLATES` definition (lines 32-74) with the following. Add the necessary Pydantic imports.

At the top of the file, after line 28 (the last existing import), add:

```python
from alphaquant.models.competitor import CompetitorAnalysis
from alphaquant.models.risk import RiskAssessment
from alphaquant.models.valuation import ValuationResult
from alphaquant.models.report import InvestmentReport
from pydantic import BaseModel
```

Then replace `_TASK_TEMPLATES` (lines 32-74) with:

```python
# Sub-project 3: each entry is (role_key, description_template, output_pydantic_model_or_None).
# Data tasks (idx 0-3) keep None (their tool returns raw JSON, parsed in Flow).
# Analysis tasks (idx 4-6) get Pydantic models. Report writer (idx 7) gets InvestmentReport.
_TASK_TEMPLATES: list[tuple[str, str, type[BaseModel] | None]] = [
    (
        "company_resolver",
        "Validate ticker '{ticker}' and return canonical company metadata.",
        None,
    ),
    (
        "market_analyst",
        "Fetch market data for '{ticker}'.",
        None,
    ),
    (
        "news_analyst",
        "Fetch recent news for '{ticker}'.",
        None,
    ),
    (
        "financial_analyst",
        "Fetch financial statements for '{ticker}'.",
        None,
    ),
    (
        "competitor_analyst",
        "Identify competitors and compute competitive score for '{ticker}'.",
        CompetitorAnalysis,
    ),
    (
        "risk_analyst",
        "Compute risk assessment for '{ticker}' from upstream data.",
        RiskAssessment,
    ),
    (
        "valuation_analyst",
        "Compute valuation (DCF + relative) for '{ticker}'.",
        ValuationResult,
    ),
    (
        "report_writer",
        "Synthesize InvestmentReport for '{ticker}'.",
        InvestmentReport,
    ),
]
```

In the `AnalysisCrew` class body (around line 86, after `def __init__`), add the class-level constant **before** `_build_tasks`:

```python
# Indices of tasks that run in parallel via async_execution=True.
# Data (0-3) and analysis (4-6) are independent → parallel.
# Report writer (7) depends on analysis outputs → serial.
_ASYNC_TASK_INDICES: set[int] = {0, 1, 2, 3, 4, 5, 6}
```

Then **replace** the `_build_tasks` method body (lines 104-122) with:

```python
def _build_tasks(self) -> list[Task]:
    tasks: list[Task] = []
    for idx, (role_key, description, pydantic_model) in enumerate(_TASK_TEMPLATES):
        agent = self.agents[idx]
        task_kwargs: dict[str, Any] = {
            "description": description,
            "expected_output": pydantic_model.__name__ if pydantic_model else "raw text",
            "agent": agent,
            "async_execution": idx in self._ASYNC_TASK_INDICES,
        }
        if pydantic_model is not None:
            task_kwargs["output_pydantic"] = pydantic_model
        # Report writer (idx 7) consumes the 3 analysis tasks' Pydantic outputs
        if idx == 7:
            task_kwargs["context"] = [tasks[4], tasks[5], tasks[6]]
        tasks.append(Task(**task_kwargs))
    return tasks
```

#### Step 4: Run tests to verify they pass

Run: `uv run pytest tests/test_crew.py::TestAnalysisCrew::test_task_templates_uses_3_tuple_with_pydantic_model tests/test_crew.py::TestAnalysisCrew::test_async_task_indices_cover_data_and_analysis_not_report -v`
Expected: PASS

If `_FakeLLM` is not importable from `tests.conftest`, add it. Open `tests/conftest.py` and add (or extend an existing class):

```python
from crewai.llm import LLM as _CrewLLM


class _FakeLLM(_CrewLLM):
    """Stub LLM that records calls and returns deterministic text. Used in unit tests."""

    def __init__(self) -> None:
        super().__init__(model="fake/model", api_key="fake")
        self.calls: list[dict[str, Any]] = []

    def call(self, messages, *args, **kwargs):  # type: ignore[override]
        from crewai.utilities.converter import ConverterError
        self.calls.append({"messages": messages, "kwargs": kwargs})
        # If a response_format (Pydantic schema) is requested, return a fake JSON string
        # that satisfies the schema. This avoids hitting the network.
        response_format = kwargs.get("response_format")
        if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            import json
            try:
                instance = response_format.model_construct()
                return instance.model_dump_json()
            except Exception:
                return "{}"
        return "fake llm response"
```

If `tests/conftest.py` already has a `_FakeLLM`, do not duplicate — adjust the test to import the existing one. If it lives elsewhere (e.g. `tests/_fake_llm.py`), update the import in the new tests accordingly.

#### Step 5: Add failing test for ReportWriter `context` parameter

In `tests/test_crew.py`, add:

```python
def test_report_writer_task_has_context_with_analysis_tasks(self):
    """Report writer (idx 7) must receive task 4/5/6 as context."""
    from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
    from unittest.mock import patch
    from tests.conftest import _FakeLLM
    fake = _FakeLLM()
    with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
        crew = _AC()
    report_task = crew.tasks[7]
    ctx = getattr(report_task, "context", None) or []
    assert len(ctx) == 3
    # context should reference the same task objects as tasks 4/5/6
    assert crew.tasks[4] in ctx
    assert crew.tasks[5] in ctx
    assert crew.tasks[6] in ctx
```

#### Step 6: Run test to verify it passes (already wired in Step 3)

Run: `uv run pytest tests/test_crew.py::TestAnalysisCrew::test_report_writer_task_has_context_with_analysis_tasks -v`
Expected: PASS

#### Step 7: Add failing tests for `output_pydantic` wiring on the 4 Pydantic tasks

In `tests/test_crew.py`, add:

```python
def test_competitor_task_has_output_pydantic(self):
    from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
    from alphaquant.models.competitor import CompetitorAnalysis
    from unittest.mock import patch
    from tests.conftest import _FakeLLM
    fake = _FakeLLM()
    with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
        crew = _AC()
    assert getattr(crew.tasks[4], "output_pydantic", None) is CompetitorAnalysis


def test_risk_task_has_output_pydantic(self):
    from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
    from alphaquant.models.risk import RiskAssessment
    from unittest.mock import patch
    from tests.conftest import _FakeLLM
    fake = _FakeLLM()
    with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
        crew = _AC()
    assert getattr(crew.tasks[5], "output_pydantic", None) is RiskAssessment


def test_valuation_task_has_output_pydantic(self):
    from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
    from alphaquant.models.valuation import ValuationResult
    from unittest.mock import patch
    from tests.conftest import _FakeLLM
    fake = _FakeLLM()
    with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
        crew = _AC()
    assert getattr(crew.tasks[6], "output_pydantic", None) is ValuationResult


def test_report_writer_task_has_output_pydantic(self):
    from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
    from alphaquant.models.report import InvestmentReport
    from unittest.mock import patch
    from tests.conftest import _FakeLLM
    fake = _FakeLLM()
    with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
        crew = _AC()
    assert getattr(crew.tasks[7], "output_pydantic", None) is InvestmentReport
```

#### Step 8: Run tests to verify they pass

Run: `uv run pytest tests/test_crew.py -v -k "output_pydantic or task_templates or async_task_indices or report_writer_task_has_context"`
Expected: 6 tests PASS (the 4 output_pydantic tests + the 2 from Step 1 + the context test from Step 5).

#### Step 9 (conditional): Widen `ValuationResult.method` Literal if needed

Before proceeding to backtory rewrites, smoke-check that the current `ValuationResult.method` Literal is wide enough. Run this Python script:

```bash
uv run python -c "
from alphaquant.models.valuation import ValuationResult
import inspect
# Read the source to find the Literal type
src = inspect.getsource(ValuationResult)
print(src)
"
```

Look for `method: Literal[...]`. The current Literal is `["dcf_relative_peg", "relative_only"]`.

If you can confirm in Task 4 (end-to-end smoke) that the LLM consistently produces only these two values, do NOTHING in this step. If the LLM produces a different value (e.g. `"dcf_only"`, `"manual"`, `"llm_estimate"`), widen the Literal in `src/alphaquant/models/valuation.py` to include the LLM's natural values. Add new values one at a time after observing them.

**If** you widen, add a unit test in `tests/test_valuation_model.py` (or the existing test file for valuation) verifying the new Literal values are accepted:

```python
def test_valuation_result_method_accepts_widened_literal():
    from alphaquant.models.valuation import ValuationResult
    v = ValuationResult(
        ticker="AAPL",
        intrinsic_value_per_share=Decimal("150.00"),
        current_price=Decimal("180.00"),
        upside_pct=-16.67,
        dcf_value=Decimal("120.00"),
        relative_value=Decimal("180.00"),
        peg_ratio=None,
        method="dcf_only",  # the new value, e.g. this one
    )
    assert v.method == "dcf_only"
```

(Adjust `method=` value to match the actual LLM output observed.)

#### Step 10: Rewrite 4 analysis agent backtories to enforce Pydantic completeness

In each of the 4 files below, **replace** the `backstory=...` argument of the `Agent(...)` constructor. Do NOT change role, goal, tools, llm, allow_delegation, or verbose.

`src/alphaquant/agents/competitor_analyst.py` — replace `backstory=(...)` with:

```python
        backstory=(
            "You are a sell-side equity analyst. You MUST call competitor_lookup with the "
            "ticker, then output a Pydantic CompetitorAnalysis object. All fields are "
            "required: target_ticker, competitors (1-10 entries), industry_rank, "
            "industry_size, competitive_score (0-100), strengths (≥1), weaknesses (≥1), "
            "method. competitors must include ticker, name, market_cap, revenue_ttm, "
            "revenue_growth_yoy, gross_margin, net_margin, pe_ratio, ps_ratio for each peer. "
            "strengths and weaknesses are short qualitative bullets derived from the metrics."
        ),
```

`src/alphaquant/agents/risk_analyst.py` — replace `backstory=(...)` with:

```python
        backstory=(
            "You are a senior risk officer. You MUST output a Pydantic RiskAssessment "
            "object. You are forbidden from omitting any of the 6 risk categories: "
            "financial, operational, market, regulatory, governance, macro. Each "
            "RiskScore entry must have category, score (0-10), rationale (≥10 chars), "
            "evidence (list of strings). total_score is 0-100 and level is one of "
            "'low', 'medium', 'high', 'extreme'. top_risks lists up to 5 short risk "
            "summaries. method is 'weighted_sum_v1'."
        ),
```

`src/alphaquant/agents/valuation_analyst.py` — replace `backstory=(...)` with:

```python
        backstory=(
            "You are a sell-side equity valuation modeler. You call the DCF tool with "
            "explicit assumptions (growth rate, WACC, terminal growth), then output a "
            "Pydantic ValuationResult. All fields required: ticker, intrinsic_value_per_share, "
            "current_price, upside_pct (%), dcf_value, relative_value, peg_ratio (nullable), "
            "method (one of the allowed Literal values), assumptions (dict of inputs you used). "
            "intrinsic_value_per_share is your blended estimate. If DCF is unavailable, use "
            "relative-only and explain in assumptions. dcf_value may be null only if FCF<=0."
        ),
```

`src/alphaquant/agents/report_writer.py` — replace `backstory=(...)` with:

```python
        backstory=(
            "You are a senior investment writer. You read the upstream CompetitorAnalysis, "
            "RiskAssessment, and ValuationResult from your context. You MUST output a Pydantic "
            "InvestmentReport object. All fields required: report_id (uuid4 string), ticker, "
            "generated_at (current ISO datetime), data_as_of (dict of source→ISO datetime), "
            "company, market, financial, financial_health_score (0-100), news, competitors, "
            "risk, valuation, rating (one of 'Strong Buy'|'Buy'|'Hold'|'Sell'|'Strong Sell'), "
            "confidence (0-100), investment_horizon ('short'|'medium'|'long'), catalysts "
            "(≥1 short bullet), markdown (≥100 chars, structured sections), sources (list of "
            "non-empty strings), disclaimer (Chinese, '本报告仅供参考，不构成投资建议...'). "
            "rating and confidence must reflect the actual risk and valuation signals, not "
            "a fixed formula."
        ),
```

#### Step 11: Run full crew tests to verify nothing regressed

Run: `uv run pytest tests/test_crew.py tests/test_agents.py -q`
Expected: All existing tests still PASS. The 4 analysis agent test_backstory_string tests (if any) may need updating — if `tests/test_agents.py` has tests like `test_competitor_analyst_backstory_mentions_competitor_lookup`, those should still pass because the new backtory still mentions the tool. If any test asserts the **exact** backtory string, update it to match the new wording (find/replace the old wording in the test).

#### Step 12: Commit

```bash
git add src/alphaquant/crews/analysis_crew.py \
        src/alphaquant/agents/competitor_analyst.py \
        src/alphaquant/agents/risk_analyst.py \
        src/alphaquant/agents/valuation_analyst.py \
        src/alphaquant/agents/report_writer.py \
        tests/test_crew.py
git commit -m "feat(crew): add output_pydantic to 4 analysis tasks + async indices

Sub-project 3: extend _TASK_TEMPLATES to 3-tuple (key, description, pydantic_model).
4 data tasks (idx 0-3) keep None (tool JSON parsed in Flow).
3 analysis tasks (idx 4-6) and report writer (idx 7) get Pydantic models
(CompetitorAnalysis, RiskAssessment, ValuationResult, InvestmentReport).

_ASYNC_TASK_INDICES lifted to class constant = {0..6}; report writer (idx 7) is
serial and receives context=[tasks[4..6]] so it sees the 3 Pydantic analysis
outputs.

4 analysis agent backtories rewritten to enforce Pydantic completeness.
Addresses sub-2 Minor M1 (_ASYNC_TASK_INDICES class constant)."
```

---

### Task 2: `parse_crew_output` Pydantic extraction + delete deterministic fallback + simplify `synthesize_report` + delete 3 scoring modules

**Files:**
- Modify: `src/alphaquant/flows/analysis_flow.py:225-309` (rewrite `parse_crew_output` body)
- Modify: `src/alphaquant/flows/analysis_flow.py:355-463` (delete `_populate_competitor` / `_populate_risk` / `_populate_valuation` / `_default_risk_subscores`)
- Modify: `src/alphaquant/flows/analysis_flow.py:534-590` (rewrite `synthesize_report` to skip rating/confidence calculation)
- Modify: `src/alphaquant/scoring/__init__.py` (remove 3 module exports)
- Delete: `src/alphaquant/scoring/rating.py`
- Delete: `src/alphaquant/scoring/competitive.py`
- Delete: `src/alphaquant/scoring/risk_score.py`
- Modify: `tests/test_flow.py` (rewrite `TestParseCrewOutput` and `TestValuationAnalysis` for Pydantic path; add tests for missing fields; delete tests that asserted deterministic fallback)
- Modify: `tests/test_scoring.py` (delete tests for removed scoring modules; keep dcf + financial_health tests)
- Test: `tests/test_flow.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `CrewOutput.tasks_output` (each `task_out` MUST have `.pydantic` attribute set by CrewAI 0.203.2 when `output_pydantic=...` is used)
- Produces:
  - `_extract_pydantic_field(tasks_output, idx, key, model_cls, state) -> BaseModel | None` — new helper
  - `parse_crew_output(result, state) -> dict[str, Any]` — extended signature (unchanged); body rewritten to use Pydantic extraction for analysis fields
  - `synthesize_report(self) -> None` — simplified; only fills `state.report.sources`, `disclaimer`, `generated_at`

---

#### Step 1: Write failing test for `_extract_pydantic_field` success path

In `tests/test_flow.py`, **add** a new test class (or extend an existing one) for the helper:

```python
class TestExtractPydanticField:
    def test_returns_pydantic_instance_from_task_output(self):
        """When task_out.pydantic is the expected model, return it directly."""
        from alphaquant.flows.analysis_flow import _extract_pydantic_field
        from alphaquant.models.competitor import CompetitorAnalysis
        from alphaquant.flows.analysis_flow import AnalysisState

        ca = CompetitorAnalysis(
            target_ticker="AAPL",
            competitors=[],
            industry_rank=1,
            industry_size=5,
            competitive_score=50,
            strengths=["x"],
            weaknesses=["y"],
            method="gics",
        )

        class _FakeTask:
            pydantic = ca
            raw = ""

        state = AnalysisState(ticker="AAPL")
        result = _extract_pydantic_field([_FakeTask(), _FakeTask()], 0, "competitor_analyst", CompetitorAnalysis, state)
        assert result is ca
        assert state.errors == []
```

#### Step 2: Run test to verify it fails

Run: `uv run pytest tests/test_flow.py::TestExtractPydanticField -v`
Expected: FAIL with `ImportError: cannot import name '_extract_pydantic_field'`

#### Step 3: Implement `_extract_pydantic_field` helper

In `src/alphaquant/flows/analysis_flow.py`, after the existing `_extract_data_field` helper (around line 324), **add** the new helper:

```python
def _extract_pydantic_field(
    tasks_output: list[Any],
    idx: int,
    key: str,
    model_cls: type[BaseModel],
    state: "AnalysisState",
) -> BaseModel | None:
    """Extract a Pydantic model from a CrewAI task output.

    CrewAI 0.203.2 sets ``task_out.pydantic`` to the validated model instance when
    the task is configured with ``output_pydantic=...``. Per sub-3 decision
    (完全删除 fallback), we ONLY read that attribute. If it is missing or not
    the expected model type, append "<key>_unavailable" to state.errors and
    return None. We do NOT attempt to recover by parsing task_out.raw.

    Returns the model instance, or ``None`` on any failure.
    """
    if idx >= len(tasks_output):
        state.errors.append(f"{key}_unavailable")
        return None
    task_out = tasks_output[idx]

    pyd_obj = getattr(task_out, "pydantic", None)
    if isinstance(pyd_obj, model_cls):
        return pyd_obj

    state.errors.append(f"{key}_unavailable")
    return None
```

Add `from pydantic import BaseModel, Field` at the top imports (already there at line 30 — verify). No new pydantic imports needed (we deliberately do not catch ValidationError since we don't validate).

#### Step 4: Run test to verify it passes

Run: `uv run pytest tests/test_flow.py::TestExtractPydanticField -v`
Expected: PASS

#### Step 5: Write failing test for Pydantic extraction in `parse_crew_output` (analysis fields)

In `tests/test_flow.py`, **add** these tests to the existing `TestParseCrewOutput` class (or create it if not present):

```python
def test_parse_crew_output_extracts_competitor_from_pydantic():
    from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
    from alphaquant.models.competitor import CompetitorAnalysis
    from alphaquant.models.company import Company
    from alphaquant.models.market import MarketData
    from alphaquant.models.financial import FinancialStatements
    from alphaquant.models.news import NewsAnalysis
    from alphaquant.models.risk import RiskAssessment
    from alphaquant.models.valuation import ValuationResult
    from alphaquant.models.report import InvestmentReport
    from decimal import Decimal

    ca = CompetitorAnalysis(
        target_ticker="AAPL", competitors=[], industry_rank=1, industry_size=5,
        competitive_score=50, strengths=["x"], weaknesses=["y"], method="gics",
    )
    company = Company(
        ticker="AAPL", name="Apple Inc.", exchange="NASDAQ",
        sector="Technology", industry="Consumer Electronics", market_cap=3_000_000_000_000,
    )
    market = MarketData(
        ticker="AAPL", price=Decimal("180"), market_cap=3_000_000_000_000,
        pe_ratio=28.0, revenue_growth_yoy=5.0, beta=1.2, source="yahoo",
        as_of=__import__("datetime").datetime.utcnow(),
    )
    fin = FinancialStatements(ticker="AAPL")
    news = NewsAnalysis.empty("AAPL")
    risk = RiskAssessment(
        ticker="AAPL", total_score=50, level="medium",
        sub_scores=[], top_risks=[], method="weighted_sum_v1",
    )
    val = ValuationResult(
        ticker="AAPL", intrinsic_value_per_share=Decimal("150"),
        current_price=Decimal("180"), upside_pct=-16.67,
        dcf_value=Decimal("120"), relative_value=Decimal("180"),
        peg_ratio=None, method="dcf_relative_peg", assumptions={},
    )
    rep = InvestmentReport(
        report_id="00000000-0000-0000-0000-000000000000", ticker="AAPL",
        generated_at=__import__("datetime").datetime.utcnow(), data_as_of={},
        company=company, market=market, financial=fin, financial_health_score=70,
        news=news, competitors=ca, risk=risk, valuation=val, rating="Hold",
        confidence=70, investment_horizon="medium", catalysts=["Earnings beat"],
        markdown="## Summary\nTest report.", sources=["yahoo"], disclaimer="本报告仅供参考。",
    )

    class _FakeTask:
        def __init__(self, pyd_obj=None, raw=""):
            self.pydantic = pyd_obj
            self.raw = raw

    tasks_output = [
        _FakeTask(pyd_obj=company, raw=company.model_dump_json()),  # 0
        _FakeTask(pyd_obj=market, raw=market.model_dump_json()),    # 1
        _FakeTask(raw='[]'),                                          # 2 (news list)
        _FakeTask(pyd_obj=fin, raw=fin.model_dump_json()),           # 3
        _FakeTask(pyd_obj=ca, raw=ca.model_dump_json()),             # 4
        _FakeTask(pyd_obj=risk, raw=risk.model_dump_json()),         # 5
        _FakeTask(pyd_obj=val, raw=val.model_dump_json()),           # 6
        _FakeTask(pyd_obj=rep, raw=rep.model_dump_json()),           # 7
    ]

    class _FakeResult:
        pass
    _FakeResult.tasks_output = tasks_output

    state = AnalysisState(ticker="AAPL")
    parse_crew_output(_FakeResult(), state)
    assert state.competitor is ca
    assert state.risk is risk
    assert state.valuation is val
    assert state.report is rep


def test_parse_crew_output_missing_pydantic_sets_none_and_appends_error():
    """When a Pydantic task output is empty, state.<field> = None + error appended."""
    from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
    from alphaquant.models.company import Company
    from alphaquant.models.market import MarketData
    from alphaquant.models.financial import FinancialStatements

    company = Company(
        ticker="AAPL", name="Apple Inc.", exchange="NASDAQ",
        sector="Technology", industry="Consumer Electronics", market_cap=3_000_000_000_000,
    )
    market = MarketData(
        ticker="AAPL", price=__import__("decimal").Decimal("180"),
        market_cap=3_000_000_000_000, pe_ratio=28.0, revenue_growth_yoy=5.0,
        beta=1.2, source="yahoo", as_of=__import__("datetime").datetime.utcnow(),
    )
    fin = FinancialStatements(ticker="AAPL")

    class _FakeTask:
        def __init__(self, pyd_obj=None, raw=""):
            self.pydantic = pyd_obj
            self.raw = raw

    tasks_output = [
        _FakeTask(pyd_obj=company, raw=company.model_dump_json()),
        _FakeTask(pyd_obj=market, raw=market.model_dump_json()),
        _FakeTask(raw="[]"),
        _FakeTask(pyd_obj=fin, raw=fin.model_dump_json()),
        _FakeTask(raw=""),  # competitor failed → empty
        _FakeTask(raw=""),  # risk failed
        _FakeTask(raw=""),  # valuation failed
        _FakeTask(raw=""),  # report writer failed
    ]

    class _FakeResult:
        pass
    _FakeResult.tasks_output = tasks_output

    state = AnalysisState(ticker="AAPL")
    parse_crew_output(_FakeResult(), state)
    assert state.competitor is None
    assert state.risk is None
    assert state.valuation is None
    assert state.report is None
    assert "competitor_analyst_unavailable" in state.errors
    assert "risk_analyst_unavailable" in state.errors
    assert "valuation_analyst_unavailable" in state.errors
    assert "report_writer_unavailable" in state.errors
```

#### Step 6: Run tests to verify they fail

Run: `uv run pytest tests/test_flow.py -v -k "extracts_competitor_from_pydantic or missing_pydantic_sets_none"`
Expected: FAIL — current `parse_crew_output` calls `_populate_competitor` etc. (deterministic fallback) and ignores the Pydantic attribute.

#### Step 7: Rewrite `parse_crew_output` body for Pydantic extraction

In `src/alphaquant/flows/analysis_flow.py`, **replace** the body of `parse_crew_output` (the part after the 4 data-field extraction block, including the 3 lines that call `_populate_competitor` / `_populate_risk` / `_populate_valuation`). The new body uses `_extract_pydantic_field` for the 4 analysis/report fields. Keep the data-field extraction unchanged.

Find the block (lines 305-309, currently calling `_safe_parse`):

```python
    # --- Competitor / Risk / Valuation: deterministic fallback (unchanged from sub-1) ---
    _populate_competitor(state, _safe_parse(raw_by_key.get("competitor_analyst", "")))
    _populate_risk(state, _safe_parse(raw_by_key.get("risk_analyst", "")))
    _populate_valuation(state, _safe_parse(raw_by_key.get("valuation_analyst", "")))

    return extracted
```

Replace it with:

```python
    # --- Sub-project 3: 3 analysis fields + 1 report from Pydantic output_pydantic ---
    state.competitor = _extract_pydantic_field(
        tasks_output, 4, "competitor_analyst", CompetitorAnalysis, state
    )
    state.risk = _extract_pydantic_field(
        tasks_output, 5, "risk_analyst", RiskAssessment, state
    )
    state.valuation = _extract_pydantic_field(
        tasks_output, 6, "valuation_analyst", ValuationResult, state
    )
    state.report = _extract_pydantic_field(
        tasks_output, 7, "report_writer", InvestmentReport, state
    )

    return extracted
```

Note: the existing 4-data-field extraction block stays unchanged. The only change is the replacement of the 3 `_populate_*` calls with the 4 `_extract_pydantic_field` calls.

#### Step 8: Run tests to verify they pass

Run: `uv run pytest tests/test_flow.py -v -k "extracts_competitor_from_pydantic or missing_pydantic_sets_none"`
Expected: PASS

#### Step 9: Delete the 4 deterministic-fallback functions + `_safe_parse` helper

In `src/alphaquant/flows/analysis_flow.py`, **delete** these 5 functions entirely (their definitions, not just the call sites — the call sites were already replaced in Step 7):

1. `_default_risk_subscores` (around lines 195-222) — replaced by LLM-driven RiskAssessment
2. `_populate_competitor` (around lines 355-387) — replaced by Pydantic extraction
3. `_populate_risk` (around lines 390-415) — replaced by Pydantic extraction
4. `_populate_valuation` (around lines 418-463) — replaced by Pydantic extraction
5. **`_safe_parse` (around line 312)** — per sub-3 "完全删除 fallback" decision, this helper has no remaining callers after Step 7. Delete it.

To find them, search for `def _default_risk_subscores`, `def _populate_competitor`, `def _populate_risk`, `def _populate_valuation`, `def _safe_parse` in the file and delete from each `def` to its closing `return` (or end of function). Also remove any helper-only imports they introduced (e.g. `from alphaquant.scoring import competitive, risk_score` at the top).

Check that the imports at the top of the file (around lines 47-48) become:

```python
from alphaquant.scoring import financial_health  # keep for backward compat (used by scoring tests)
from alphaquant.scoring.rating import determine_rating  # DELETE this line
```

(If `determine_rating` is no longer used anywhere in the file, drop the import. The other 2 imports — `financial_health` and `risk_score` from `alphaquant.scoring` — may be used elsewhere in the file; check before deleting.)

Specifically: the import line `from alphaquant.scoring.rating import determine_rating` should be removed. The `from alphaquant.scoring import financial_health, risk_score` line should be reduced to `from alphaquant.scoring import financial_health` (since `risk_score` is no longer used). The `from alphaquant.scoring.risk_score import compute` line (if any) should also be removed.

#### Step 10: Rewrite `synthesize_report` to skip rating/confidence calculation

In `src/alphaquant/flows/analysis_flow.py`, find `synthesize_report` (around lines 534-590) and **replace** its body. The new body does NOT call `determine_rating` or compute any `confidence` formula — it just fills runtime-only fields and raises on missing report.

Find the existing function and replace it with:

```python
@listen(run_crew)
def synthesize_report(self) -> None:
    """Sub-project 3: state.report is already populated by ReportWriter agent.

    This step fills only runtime-only fields (sources, disclaimer, generated_at)
    and raises ReportGenerationError if the report writer failed.
    """
    if self.state.report is None:
        log.error("report_writer_failed", ticker=self.state.ticker)
        raise ReportGenerationError(
            f"Report writer agent failed to produce InvestmentReport for {self.state.ticker}"
        )

    # Re-derive sources from upstream data (so they reflect actual data presence, not LLM guess)
    self.state.report.sources = _collect_sources(
        self.state.market, self.state.news, self.state.financial, self.state.competitor
    )

    # Runtime fields
    self.state.report.disclaimer = DISCLAIMER_TEXT  # constant from sub-1, kept verbatim
    self.state.report.generated_at = datetime.utcnow()
```

The `DISCLAIMER_TEXT` constant already exists in the file (used by the old `synthesize_report`). Do not change it.

#### Step 11: Delete the 3 unused scoring modules

```bash
git rm src/alphaquant/scoring/rating.py
git rm src/alphaquant/scoring/competitive.py
git rm src/alphaquant/scoring/risk_score.py
```

In `src/alphaquant/scoring/__init__.py`, **replace** the entire body with:

```python
"""Scoring module — only deterministic helpers used by tools and LLM tools remain.

Sub-project 3: `rating`, `competitive`, `risk_score` modules removed (LLM-driven now).
`dcf` and `financial_health` remain because the ValuationAnalyst and ReportWriter
agents can call them as tools during reasoning.
"""
from alphaquant.scoring import dcf, financial_health

__all__ = ["dcf", "financial_health"]
```

#### Step 12: Delete the now-broken tests in `tests/test_scoring.py`

In `tests/test_scoring.py`, find and **delete** all test classes/functions that import from the removed modules:

- `TestDetermineRating` (or similar) — tests for `scoring.rating.determine_rating`
- `TestCompetitive` (or similar) — tests for `scoring.competitive.compute`
- `TestRiskScore` (or similar) — tests for `scoring.risk_score.compute` / `determine_level`

To find them, search the file for `from alphaquant.scoring.rating`, `from alphaquant.scoring.competitive`, `from alphaquant.scoring.risk_score` and delete the corresponding `class` / `def` blocks.

**Keep** tests for `scoring.dcf.compute_dcf_value` (sub-1) and `scoring.financial_health.compute` (sub-1).

#### Step 13: Delete obsolete tests in `tests/test_flow.py` and `tests/test_agents.py`

In `tests/test_flow.py`, find and **delete** tests that asserted the deterministic fallback behavior. Search for these test names (they may differ slightly — search by content):

- `test_competitor_fallback_gics` or any test asserting `_populate_competitor` was called
- `test_risk_default_subscores` or any test asserting `_default_risk_subscores` was used
- `test_valuation_deterministic` or any test asserting `_populate_valuation` was called
- `test_synthesize_report_computes_confidence` or any test asserting the formula path

To find them, grep the file for `confidence`, `data_completeness`, `method_coverage`, `signal_alignment`, `_default_risk_subscores`, `_populate_competitor`, `_populate_risk`, `_populate_valuation`. Any test that exercises these deleted functions or asserts the old formula's output should be deleted or rewritten.

**Do NOT delete**:
- The 4 data-field extraction tests (still valid in sub-3, unchanged behavior)
- The `test_company_fetch_failure_raises_all_sources_down` test (still valid)
- The flow timeout test (will be updated in Task 3 to expect 600s)

In `tests/test_agents.py`, find any test that imports `from alphaquant.scoring.rating import determine_rating` or similar and delete those tests.

#### Step 14: Run full test suite

Run: `uv run pytest tests/ -q`
Expected: 224 - (removed tests) + (added tests) passing. Net should be at least 224 (regressions not allowed) and at most ~230.

If some old tests still reference deleted symbols, fix them per Step 13 (delete or rewrite).

#### Step 15: Commit

```bash
git add src/alphaquant/flows/analysis_flow.py \
        src/alphaquant/scoring/__init__.py \
        src/alphaquant/scoring/rating.py \
        src/alphaquant/scoring/competitive.py \
        src/alphaquant/scoring/risk_score.py \
        tests/test_flow.py \
        tests/test_scoring.py \
        tests/test_agents.py
git commit -m "feat(flow): parse_crew_output uses Pydantic output; remove fallback

Sub-project 3: parse_crew_output now extracts competitor/risk/valuation/report
via _extract_pydantic_field (new helper) from CrewAI's task_out.pydantic
attribute. _populate_competitor / _populate_risk / _populate_valuation /
_default_risk_subscores deleted.

synthesize_report simplified: state.report is already populated by
ReportWriter agent; this step only fills runtime fields (sources,
disclaimer, generated_at) and raises ReportGenerationError if report
writer failed.

scoring/rating.py, scoring/competitive.py, scoring/risk_score.py removed
(LLM-driven now). scoring/dcf.py and scoring/financial_health.py remain
(used as tools by ValuationAnalyst and ReportWriter)."
```

---

### Task 3: Fix 3 deferred blockers — sync kickoff, timeout, tool empty-shell

**Files:**
- Modify: `src/alphaquant/flows/analysis_flow.py:54` (FLOW_TIMEOUT_SECONDS 180 → 600)
- Modify: `src/alphaquant/flows/analysis_flow.py:108-200` (rewrite `run_crew` with sync kickoff wrap, IF Step 5.5 shows Blocker 1 still occurs)
- Modify: `src/alphaquant/tools/company_lookup_tool.py` (drop `except Exception` empty-shell fallback)
- Modify: `src/alphaquant/tools/market_data_tool.py` (drop `except Exception` empty-shell fallback)
- Modify: `src/alphaquant/tools/news_tool.py` (drop `except Exception` empty-shell fallback)
- Modify: `src/alphaquant/tools/financial_tool.py` (drop `except Exception` empty-shell fallback)
- Modify: `tests/test_flow.py` (update flow timeout test; add test for sync kickoff IF needed)
- Modify: `tests/test_tools.py` (update tool tests to expect error string on exception, not empty shell)
- Test: `tests/test_flow.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `AnalysisFlow.state`, `AnalysisCrew.kickoff(inputs: dict)`
- Produces:
  - `FLOW_TIMEOUT_SECONDS: float = 600.0` (module constant in `analysis_flow.py`)
  - `AnalysisFlow.run_crew(self) -> None` — uses `asyncio.wait_for(asyncio.to_thread(self._kickoff_sync), timeout=FLOW_TIMEOUT_SECONDS)` *(only if Step 5.5 shows Blocker 1 still occurs)*
  - `AnalysisFlow._kickoff_sync(self) -> CrewOutput` — new private method *(conditional on Step 5.5)*

**Pre-Flight Gate (revised 2026-06-21 after Step 0 partial reproduction):**

The original plan assumed Blocker 1 (asyncio shutdown race in `parse_crew_output`) was the proximate issue. **Step 0 partial reproduction (commit `dbfac17`'s evidence, log `/tmp/sub3-blocker1-aapl.log`) showed this assumption is wrong.** The actual proximate blocker is **Blocker 2** (`FLOW_TIMEOUT_SECONDS=180`) — AAPL hits the 180s timeout before `parse_crew_output` is ever called. Blocker 1 may or may not still occur after Blocker 2 is fixed; we don't know yet.

Also discovered during Step 0 (committed as `dbfac17` BEFORE Task 3 began):
- `ValuationResult.method` Literal too narrow — LLM produces "blended"
- `CompetitorAnalysis.method` Literal too narrow — LLM produces "hybrid"
- Both Literal constraints have been widened in `dbfac17` with tests in `tests/test_models_literals.py`

**Revised task order:**
1. Step 0 (already partial-done): reproduce and document what blocks first
2. **Steps 1-5 first**: widen `FLOW_TIMEOUT_SECONDS` 180→600 (Blocker 2 fix)
3. **New Step 5.5**: re-run AAPL with widened timeout + widened Literals to actually attempt Blocker 1 reproduction
4. **Steps 6-10**: ONLY IF Step 5.5 shows Blocker 1 still occurs — sync kickoff wrap (Blocker 1 fix)
5. **Steps 11-16**: tool empty-shell fallback (Blocker 3 fix) — independent of 1+2
6. **Step 17**: re-verify all blockers fixed end-to-end

---

#### Step 0: Reproduce blockers end-to-end (PARTIAL — already done in commit `dbfac17`'s evidence)

The first implementer run of this step produced these findings (saved at `/tmp/sub3-blocker1-aapl.log`, 48KB). **Reproduce to verify they still hold**, then document:

```bash
# Pre-flight: API key + model must be correct
grep MINIMAX_API_KEY .env     # must NOT be placeholder
grep LITELLM_MODEL .env       # must be openai/MiniMax-M3

# Run for 30s and observe (don't wait full 600s — we just want to confirm what blocks first)
timeout 600 uv run python -m alphaquant AAPL --format json 2>&1 | tee /tmp/sub3-blocker1-aapl-step0.log | tail -20
```

**Expected observation** (per the existing log `/tmp/sub3-blocker1-aapl.log`):
- AAPL hits `flow_timeout ticker=AAPL timeout_seconds=180.0` BEFORE reaching `parse_crew_output`
- LLM produces `ValuationResult.method="blended"` and `CompetitorAnalysis.method="hybrid"` (now accepted post-`dbfac17`)
- CrewAI's converter may still raise `ConverterError` for some tasks due to retry-path issues

Document in `task-3-report.md` under "Sub-3 Step 0: Blocker reproduction":

```markdown
## Sub-3 Step 0 (this run, commit <commit-hash>)

### What actually blocks first
- [ ] Confirmed: 180s flow_timeout fires before parse_crew_output → Blocker 2 is proximate
- [ ] Other blockers observed: <list>

### LLM output observations
- ValuationResult.method seen: <list>
- CompetitorAnalysis.method seen: <list>
- Other Literal violations: <list>

### Blocker 1 (asyncio shutdown race)
- [ ] Still reproduces? Y/N
- If N: probably won't manifest after Blocker 2 is fixed — verify in Step 5.5
- If Y: needs sync kickoff wrap from Steps 6-10
```

**Implementation rule:** If Step 0 shows Blocker 2 is NOT the proximate blocker (i.e. `parse_crew_output` IS reached and Blocker 1 actually occurs first), STOP and report — the rest of the plan assumes Blocker 2 first.

---

#### Step 1: Write failing test for `FLOW_TIMEOUT_SECONDS == 600`

In `tests/test_flow.py`, **add** this test (or find an existing flow-timeout test and update its expected value):

```python
def test_flow_timeout_seconds_is_600():
    """Sub-3 widens timeout to 600s for real LLM latency."""
    from alphaquant.flows.analysis_flow import FLOW_TIMEOUT_SECONDS
    assert FLOW_TIMEOUT_SECONDS == 600.0
```

#### Step 2: Run test to verify it fails

Run: `uv run pytest tests/test_flow.py::test_flow_timeout_seconds_is_600 -v`
Expected: FAIL — current value is 180.0.

#### Step 3: Update `FLOW_TIMEOUT_SECONDS` to 600

In `src/alphaquant/flows/analysis_flow.py` line 54, change:

```python
FLOW_TIMEOUT_SECONDS = 180.0
```

to:

```python
# §3.4: whole-Flow timeout. Sub-project 3 widens 180→600s to absorb
# 4 parallel data fetches (~30s each) + 3 parallel LLM analysis tasks
# (~60-90s) + 1 ReportWriter task (~30-60s) + manager overhead. Total
# expected ~120-180s; 600s leaves 3-4x headroom.
FLOW_TIMEOUT_SECONDS = 600.0
```

#### Step 4: Run test to verify it passes

Run: `uv run pytest tests/test_flow.py::test_flow_timeout_seconds_is_600 -v`
Expected: PASS

#### Step 5: Update the existing flow-timeout test (if it asserts 180)

Search `tests/test_flow.py` for any test like `test_kickoff_with_timeout_succeeds_under_limit` or `test_flow_times_out_after_seconds`. If it asserts a 180s constant or 180s timeout expectation, change to 600.

---

#### Step 5.5: Re-run AAPL with widened timeout + widened Literals — does Blocker 1 still occur?

**NEW step** inserted after Step 0's discovery (commit `dbfac17` evidence in `/tmp/sub3-blocker1-aapl.log`). We can now finally attempt to reproduce Blocker 1 (asyncio shutdown race) because:
- The timeout has been widened (Step 3) — Blocker 2 won't fire prematurely
- The Literals have been widened (commit `dbfac17`, pre-Task 3) — LLM output won't fail validation

Run:

```bash
timeout 600 uv run python -m alphaquant AAPL --format json 2>&1 | tee /tmp/sub3-step5.5-aapl.log | tail -20
```

**Three possible outcomes**, each with a different code path:

**Outcome A: AAPL succeeds and emits an `InvestmentReport` JSON.**

```bash
# Confirm: does the output look like a real report?
cat /tmp/sub3-step5.5-aapl.log | tail -1 | jq '.valuation.dcf_value, .report.rating, .report.confidence'
```

If yes: Blocker 1 doesn't exist, OR it was already fixed by the timeout widening + Literal widening. **Skip Steps 6-10 entirely** (no `_kickoff_sync` wrap needed). Document and proceed to Step 11.

**Outcome B: AAPL fails with `RuntimeError: cannot schedule new futures after shutdown` in `parse_crew_output`.**

This is the original Blocker 1 hypothesis confirmed. **Proceed with Steps 6-10** (sync kickoff wrap).

**Outcome C: AAPL fails with something else (LLM error, network error, ConverterError without `RuntimeError`, etc.).**

Debug the specific error first; treat as a NEW blocker. Document and STOP — do not proceed with Steps 6-10 (they may not address the actual cause). Possibly need to revise the plan again.

Document in `task-3-report.md` under "Sub-3 Step 5.5: Blocker 1 re-attempt":

```markdown
## Sub-3 Step 5.5 (commit <commit-hash>)

### Outcome
A / B / C

### Evidence
<log excerpt>

### Decision
- Outcome A: skipped Steps 6-10
- Outcome B: proceed with Steps 6-10
- Outcome C: new blocker discovered, see <section>
```

---

#### Step 6: Write failing test for sync kickoff wrap (Blocker 1 fix — CONDITIONAL on Step 5.5 outcome B)

In `tests/test_flow.py`, **add**:

```python
def test_run_crew_uses_kickoff_sync_wrapped_in_to_thread():
    """run_crew must wrap crew.kickoff in a sync function + asyncio.to_thread.

    This ensures asyncio.wait_for can cancel mid-execution (Blocker 1 fix).
    """
    import inspect
    from alphaquant.flows.analysis_flow import AnalysisFlow
    src = inspect.getsource(AnalysisFlow.run_crew)
    # Must contain _kickoff_sync helper
    assert "_kickoff_sync" in src
    # Must wrap with asyncio.to_thread
    assert "asyncio.to_thread" in src
    # Must use asyncio.wait_for for timeout
    assert "asyncio.wait_for" in src
```

#### Step 7: Run test to verify it fails

Run: `uv run pytest tests/test_flow.py::test_run_crew_uses_kickoff_sync_wrapped_in_to_thread -v`
Expected: FAIL — current `run_crew` directly does `asyncio.to_thread(crew.kickoff, inputs=...)` without a sync helper.

#### Step 8: Rewrite `run_crew` to use sync kickoff wrap (CONDITIONAL on Step 5.5 outcome B)

**Skip this entire step if Step 5.5 produced Outcome A** (AAPL succeeded — Blocker 1 doesn't exist).

In `src/alphaquant/flows/analysis_flow.py`, find the existing `run_crew` method (around line 108-150). **Replace** its entire body with:

```python
@start()
async def run_crew(self) -> None:
    """Kick off the CrewAI crew, parse the output, and populate state.

    Sub-project 3 (Blocker 1 fix): wrap ``crew.kickoff(inputs=...)`` in a sync
    helper function and call it via ``asyncio.to_thread``, so that
    ``asyncio.wait_for`` can cancel the future mid-execution. Calling
    ``asyncio.to_thread(crew.kickoff, ...)`` directly (sub-2 pattern) passes
    the bound method as a callable, and ``wait_for`` cancellation can race
    with the method's own event-loop teardown. The sync helper keeps the
    call stack predictable.
    """
    # Ticker normalization and validation
    self.state.ticker = _normalize_ticker(self.state.ticker)
    log.info("crew_started", ticker=self.state.ticker)

    def _kickoff_sync() -> CrewOutput:
        return AnalysisCrew().kickoff(inputs={"ticker": self.state.ticker})

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_kickoff_sync),
            timeout=FLOW_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.error("crew_timeout", ticker=self.state.ticker, timeout_seconds=FLOW_TIMEOUT_SECONDS)
        raise

    parse_crew_output(result, self.state)
```

The existing `_normalize_ticker` is invoked. If the existing `run_crew` does any additional work (state setup, observability events), preserve those — they go **before** `_kickoff_sync` definition.

Add `from alphaquant.crews import AnalysisCrew` import (likely already there at line 32 — verify). Add `from crewai.agents import CrewOutput` if not already imported (it should be — see sub-1 spec). Verify with `grep "CrewOutput" src/alphaquant/flows/analysis_flow.py` and add if missing.

#### Step 9: Run test to verify it passes

Run: `uv run pytest tests/test_flow.py::test_run_crew_uses_kickoff_sync_wrapped_in_to_thread -v`
Expected: PASS

#### Step 10: Run the full flow test suite to make sure no test broke

Run: `uv run pytest tests/test_flow.py -q`
Expected: All flow tests PASS. The only change is the `run_crew` body, which existing tests should still satisfy via the same async interface.

#### Step 11: Write failing test for tool no-empty-shell (Blocker 3 fix)

In `tests/test_tools.py`, find the existing `TestCompanyLookupTool` class and **add** a new test method (or extend an existing test that previously expected an empty shell):

```python
def test_no_empty_shell_on_unexpected_exception(self):
    """When the registry raises an unexpected exception, tool returns error string.

    Sub-project 3 (Blocker 3 fix): the old behavior returned {"name": "N/A", ...}
    empty shell. New behavior: error string, which parse_crew_output detects
    and treats as failure.
    """
    from alphaquant.tools.company_lookup_tool import CompanyLookupTool

    class FakeRegistry:
        async def get_company(self, ticker):
            raise RuntimeError("network glitch")

    with patch("alphaquant.tools.company_lookup_tool.DataSourceRegistry", FakeRegistry):
        result = CompanyLookupTool()._run("AAPL")

    # Must be an error string, NOT an empty shell dict
    assert result.startswith("Error"), f"expected 'Error...' prefix, got: {result!r}"
    assert "network glitch" in result or "RuntimeError" in result
```

Repeat the same shape for `TestMarketDataTool`, `TestNewsTool`, `TestFinancialTool` — search for any test that previously asserted an empty-shell return on unexpected exception, and update its expected value. Or add new tests if none existed.

Concretely, open each of the 4 tool test classes in `tests/test_tools.py` and add a `test_no_empty_shell_on_unexpected_exception` method that follows the pattern above (substituting the correct `DataSourceRegistry` method and `CompanyLookupTool` / `MarketDataTool` / `NewsTool` / `FinancialTool`).

#### Step 12: Run tests to verify they fail

Run: `uv run pytest tests/test_tools.py -v -k "no_empty_shell"`
Expected: FAIL — current tools return `{"name": "N/A", ...}` (or `{}`) on unexpected exception.

#### Step 13: Drop `except Exception` empty-shell fallback in 4 data tools

In each of these 4 files, find the `except Exception as e:` (or similar) block in the `_run` method and **replace** its body so it returns an error string instead of an empty shell:

`src/alphaquant/tools/company_lookup_tool.py` — find the block that returns `{"name": "N/A", ...}` or similar empty dict. Replace it with:

```python
        except Exception as e:
            return f"Error fetching company: {type(e).__name__}: {e}"
```

(Keep the `AllDataSourcesDown` handler and `asyncio.TimeoutError` handler above this block unchanged. Only modify the `except Exception` fallback.)

`src/alphaquant/tools/market_data_tool.py` — replace the `except Exception` empty fallback with:

```python
        except Exception as e:
            return f"Error fetching market data: {type(e).__name__}: {e}"
```

`src/alphaquant/tools/news_tool.py` — replace the `except Exception` empty fallback with:

```python
        except Exception as e:
            return f"Error fetching news: {type(e).__name__}: {e}"
```

`src/alphaquant/tools/financial_tool.py` — replace the `except Exception` empty fallback with:

```python
        except Exception as e:
            return f"Error fetching financials: {type(e).__name__}: {e}"
```

To locate the exact lines, search each file for `except Exception` and inspect the `return` statement. The current code (per sub-2) probably looks like:

```python
        except Exception:
            return {"name": "N/A", "ticker": ticker, "exchange": "N/A", ...}  # or similar empty shell
```

Replace it entirely with the `Error fetching X: ...` string return.

If a tool's `_run` does NOT have an `except Exception` empty-shell fallback (some may not), skip the modification for that file.

#### Step 14: Run tests to verify they pass

Run: `uv run pytest tests/test_tools.py -v -k "no_empty_shell"`
Expected: PASS

#### Step 15: Run full test suite

Run: `uv run pytest tests/ -q`
Expected: All tests PASS. The only changes are:
- `FLOW_TIMEOUT_SECONDS` 180 → 600 (existing timeout tests may need 600 update — done in Step 5)
- `run_crew` body rewritten (existing test mocks may need to mock `_kickoff_sync` instead of `crew.kickoff` — see Step 16)
- 4 tool empty-shell fallbacks dropped

If any flow test fails because it mocks `asyncio.to_thread(crew.kickoff, inputs=...)` with the sub-2 pattern, update the mock to target `_kickoff_sync` or the new `crew.kickoff` call inside it.

#### Step 16: Update any sub-2 flow test that mocks the old kickoff pattern

Search `tests/test_flow.py` for `asyncio.to_thread` and `crew.kickoff` patterns. The sub-2 test probably does something like:

```python
with patch("asyncio.to_thread") as mock_to_thread:
    ...
```

If the test relied on the old `asyncio.to_thread(crew.kickoff, ...)` shape, update it to use `patch.object(AnalysisFlow, "_kickoff_sync")` or a similar approach that targets the new helper. The flow test from sub-2 (`test_run_crew_invokes_crew_with_only_ticker` or similar) may need this fix.

#### Step 17: Re-verify Blocker 1 is fixed end-to-end

Repeat Step 0's reproduction:

```bash
timeout 600 uv run python -m alphaquant AAPL --format json 2>&1 | tee /tmp/sub3-blocker1-aapl-after.log | tail -20
```

Expected: AAPL completes successfully and emits a JSON InvestmentReport (no `INTERNAL_ERROR`, no `RuntimeError: cannot schedule new futures after shutdown`). If still failing, return to Step 8 and revise.

Document the before/after in `task-3-report.md` under a new "Sub-3 Step 17 verification" section.

#### Step 18: Commit

```bash
git add src/alphaquant/flows/analysis_flow.py \
        src/alphaquant/tools/company_lookup_tool.py \
        src/alphaquant/tools/market_data_tool.py \
        src/alphaquant/tools/news_tool.py \
        src/alphaquant/tools/financial_tool.py \
        tests/test_flow.py \
        tests/test_tools.py \
        task-3-report.md
git commit -m "fix(flow): 3 deferred blockers — sync kickoff, timeout 600, no empty shell

Sub-project 3:

- Blocker 1 (asyncio shutdown race): run_crew now wraps crew.kickoff(inputs=...)
  in _kickoff_sync() and calls it via asyncio.to_thread(), so asyncio.wait_for
  can cancel mid-execution. The sub-2 pattern of asyncio.to_thread(crew.kickoff, ...)
  passed a bound method that races with event-loop teardown.

- Blocker 2 (timeout too short): FLOW_TIMEOUT_SECONDS 180 → 600 to absorb
  4 parallel data fetches + 3 parallel LLM analysis + 1 report writer +
  manager overhead.

- Blocker 3 (tool empty-shell fallback): 4 data tools (company_lookup,
  market_data, news, financial) no longer return {name: 'N/A', ...} empty
  shells on unexpected exception. They return 'Error fetching X: ...' strings,
  which parse_crew_output's error-string detector already handles as failure.

Reproduced and verified via real LLM end-to-end (AAPL). See task-3-report.md."
```

---

### Task 4: Graceful degradation E2E + real LLM smoke

**Files:**
- Modify: `tests/test_flow.py` (add `test_unknown_ticker_raises_all_data_sources_down` integration test)
- Modify: `tests/test_observability.py` (add assertion for ZZZZZZ → AllDataSourcesDown event log)
- Test: `tests/test_flow.py`
- Test: `tests/test_observability.py`

**Note**: This task has 2 parts:
1. Unit-level test that mocks the ZZZZZZ end-to-end path through Flow → crew → tool → AllDataSourcesDown. Runs in CI.
2. Real LLM smoke test (AAPL/MSFT/TSLA/ZZZZZZ) that requires `MINIMAX_API_KEY` and a working LLM. Runs locally / manually; not gated by CI.

---

#### Step 1: Write failing test for ZZZZZZ graceful degradation

In `tests/test_flow.py`, **add** a new test class for the full-flow integration:

```python
class TestGracefulDegradation:
    def test_unknown_ticker_raises_all_data_sources_down(self):
        """ZZZZZZ (format-valid, registry-unknown) → parse_crew_output raises AllDataSourcesDown.

        Sub-3 Blocker 3 verification: end-to-end graceful degradation.
        """
        from alphaquant.exceptions import AllDataSourcesDown
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.models.market import MarketData
        from alphaquant.models.news import NewsAnalysis
        from alphaquant.models.financial import FinancialStatements
        from decimal import Decimal
        import datetime

        # Mock 4 data tools: only company_lookup returns an error string
        class _FakeTask:
            def __init__(self, pyd_obj=None, raw=""):
                self.pydantic = pyd_obj
                self.raw = raw

        company_error = "Error fetching company: AllDataSourcesDown: cannot resolve ZZZZZZ"
        market = MarketData(
            ticker="ZZZZZZ", price=Decimal("0"), market_cap=0,
            pe_ratio=None, revenue_growth_yoy=None, beta=None, source="degraded",
            as_of=datetime.datetime.utcnow(),
        )
        news = NewsAnalysis.empty("ZZZZZZ")
        fin = FinancialStatements(ticker="ZZZZZZ")

        tasks_output = [
            _FakeTask(raw=company_error),  # company_resolver failed
            _FakeTask(pyd_obj=market, raw=market.model_dump_json()),
            _FakeTask(raw="[]"),
            _FakeTask(pyd_obj=fin, raw=fin.model_dump_json()),
            _FakeTask(raw=""),
            _FakeTask(raw=""),
            _FakeTask(raw=""),
            _FakeTask(raw=""),
        ]

        class _FakeResult:
            pass
        _FakeResult.tasks_output = tasks_output

        state = AnalysisState(ticker="ZZZZZZ")
        with pytest.raises(AllDataSourcesDown) as exc_info:
            parse_crew_output(_FakeResult(), state)
        assert "ZZZZZZ" in str(exc_info.value)
```

#### Step 2: Run test to verify it passes (already implemented in sub-2; this is regression)

Run: `uv run pytest tests/test_flow.py::TestGracefulDegradation -v`
Expected: PASS — sub-2 already implemented the company-error → AllDataSourcesDown path via `_extract_data_field`. This test just locks in the behavior.

If this test fails, debug the error string detection in `_extract_data_field` (the `if raw.startswith("Error")` check should catch `"Error fetching company: ..."`).

#### Step 3: Add observability assertion for AllDataSourcesDown event

In `tests/test_observability.py`, find the test class that exercises flow events and **add**:

```python
def test_company_failure_logs_all_data_sources_down_event(caplog):
    """When company fetch fails, an AllDataSourcesDown event is logged."""
    import logging
    from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
    from alphaquant.exceptions import AllDataSourcesDown
    from alphaquant.models.market import MarketData
    from alphaquant.models.news import NewsAnalysis
    from alphaquant.models.financial import FinancialStatements
    from decimal import Decimal
    import datetime

    with caplog.at_level(logging.ERROR, logger="alphaquant.flows.analysis_flow"):
        class _FakeTask:
            def __init__(self, raw=""):
                self.raw = raw

        company_error = "Error fetching company: AllDataSourcesDown: cannot resolve ZZZZZZ"
        market = MarketData(
            ticker="ZZZZZZ", price=Decimal("0"), market_cap=0,
            pe_rating=None, revenue_growth_yoy=None, beta=None, source="degraded",
            as_of=datetime.datetime.utcnow(),
        )
        news = NewsAnalysis.empty("ZZZZZZ")
        fin = FinancialStatements(ticker="ZZZZZZ")

        tasks_output = [
            _FakeTask(raw=company_error),
            _FakeTask(pyd_obj=market, raw=market.model_dump_json()),
            _FakeTask(raw="[]"),
            _FakeTask(pyd_obj=fin, raw=fin.model_dump_json()),
            _FakeTask(raw=""),
            _FakeTask(raw=""),
            _FakeTask(raw=""),
            _FakeTask(raw=""),
        ]

        class _FakeResult:
            pass
        _FakeResult.tasks_output = tasks_output

        state = AnalysisState(ticker="ZZZZZZ")
        try:
            parse_crew_output(_FakeResult(), state)
        except AllDataSourcesDown:
            pass

        assert any("company" in r.message.lower() or "all_data_sources_down" in r.message.lower() for r in caplog.records)
```

(Tweak the assertion based on the actual log format used in `alphaquant.observability`. The sub-2 pattern likely uses structured logging with `log.error("company_fetch_failed", ticker=...)` — adjust the assertion to match.)

#### Step 4: Run the new observability test

Run: `uv run pytest tests/test_observability.py -v -k "company_failure_logs_all_data_sources_down"`
Expected: PASS

#### Step 5: Run full test suite to confirm baseline

Run: `uv run pytest tests/ -q`
Expected: 224 - (removed tests) + (added tests) passing. Target: ≥ 236. Verify the new test count is at least 12 above the sub-2 baseline (10 from Task 1 + 5 from Task 2 + 5 from Task 3 + 2 from Task 4 = 22, minus any deleted tests).

#### Step 6: Real LLM smoke test (manual, gated by `MINIMAX_API_KEY`)

This step is **not a CI test** — it's a manual verification step. The implementer should run it locally with a real API key in `.env` to confirm sub-3 end-to-end behavior. Document the results in a `task-4-report.md` file in `.superpowers/sdd/`.

Run these commands and capture their outputs:

```bash
# Pre-flight
grep MINIMAX_API_KEY .env   # must be real, not placeholder
grep LITELLM_MODEL .env     # must be openai/MiniMax-M3 (per sub-2 fix)

# Test 1: AAPL — happy path
timeout 600 uv run python -m alphaquant AAPL --format json 2>/dev/null | tail -1 | jq '.valuation.dcf_value, .report.rating, .report.confidence, .report.catalysts, .report.markdown[:200]'
# Expected: dcf_value is a number, rating is one of 5 Literal values, confidence 0-100, catalysts list non-empty, markdown is multi-line

# Test 2: MSFT — different company
timeout 600 uv run python -m alphaquant MSFT --format json 2>/dev/null | tail -1 | jq '.valuation.dcf_value, .report.rating, .report.confidence'
# Expected: different from AAPL

# Test 3: TSLA — different company
timeout 600 uv run python -m alphaquant TSLA --format json 2>/dev/null | tail -1 | jq '.valuation.dcf_value, .report.rating, .report.confidence'
# Expected: different from AAPL and MSFT

# Test 4: ZZZZZZ — graceful degradation (Blocker 3 end-to-end)
timeout 600 uv run python -m alphaquant ZZZZZZ 2>&1 | tail -3
# Expected: {"code": "ALL_DATA_SOURCES_DOWN", "message": "Cannot resolve ZZZZZZ: company data unavailable"}
# NOT: INTERNAL_ERROR or 180s timeout
```

If any of these fails:
- AAPL/MSFT/TSLA: LLM may not be producing Pydantic-valid output. Check `state.errors` via the test path; consider prompt tuning in `Task 1` backtory rewrites.
- ZZZZZZ: If the output is `INTERNAL_ERROR` instead of `ALL_DATA_SOURCES_DOWN`, debug the tool's `AllDataSourcesDown` propagation — `tools/company_lookup_tool.py` may not be re-raising the exception correctly.

#### Step 7: Write task-4 report

Create `.superpowers/sdd/task-4-report.md` with the smoke test outputs and any prompt tuning notes. Commit it (the `.superpowers/sdd/` directory is gitignored, but documenting findings is still useful for review).

```markdown
# Task 4 Report: E2E smoke + graceful degradation verification

## Test results
- AAPL: <output snippet>
- MSFT: <output snippet>
- TSLA: <output snippet>
- ZZZZZZ: <output snippet>

## Confidence comparison
- AAPL: <conf>
- MSFT: <conf>
- TSLA: <conf>

## Blockers
- <list any unresolved issues>

## Recommendation
- <ready for review / needs follow-up>
```

#### Step 8: Commit task report (if non-empty findings)

```bash
git add .superpowers/sdd/task-4-report.md
git commit -m "docs(sdd): sub-3 task 4 E2E smoke test results"
```

If no findings, skip this commit.

---

## Type Consistency

| Symbol | Type | Defined In |
|---|---|---|
| `_TASK_TEMPLATES: list[tuple[str, str, type[BaseModel] \| None]]` | 3-tuple list, length 8 | Task 1 Step 3 |
| `AnalysisCrew._ASYNC_TASK_INDICES: set[int]` | class-level constant, value `{0, 1, 2, 3, 4, 5, 6}` | Task 1 Step 3 |
| `AnalysisCrew.tasks[7].context: list[Task]` | length 3, contains tasks 4/5/6 | Task 1 Step 3 |
| `AnalysisCrew.tasks[4].output_pydantic = CompetitorAnalysis` | matches `_TASK_TEMPLATES[4][2]` | Task 1 Step 3 |
| `_extract_pydantic_field(tasks_output, idx, key, model_cls, state) -> BaseModel \| None` | new helper | Task 2 Step 3 |
| `parse_crew_output(result, state) -> dict[str, Any]` | unchanged signature, rewritten body | Task 2 Step 7 |
| `FLOW_TIMEOUT_SECONDS: float = 600.0` | module constant in `analysis_flow.py` | Task 3 Step 3 |
| `AnalysisFlow._kickoff_sync(self) -> CrewOutput` | new private method, calls `AnalysisCrew().kickoff(inputs={"ticker": self.state.ticker})` | Task 3 Step 8 |
| `synthesize_report(self) -> None` | unchanged signature, simplified body | Task 2 Step 10 |
| `scoring.rating`, `scoring.competitive`, `scoring.risk_score` | **deleted** | Task 2 Step 11 |
| `scoring.dcf`, `scoring.financial_health` | unchanged, still importable from `alphaquant.scoring` | Task 2 Step 11 |

## Risks & Trade-offs (carried from spec §Risks)

1. **MiniMax-M3 Pydantic 稳定性**: `Task(output_pydantic=...)` relies on the LLM producing schema-valid JSON. CrewAI 0.203.2 has internal retry logic (typically 1-3 attempts). If it still fails, `_extract_pydantic_field` returns `None` and `state.errors.append("<key>_unavailable")` — the flow continues. **Mitigation**: smoke test in Task 4 catches this; prompt tuning in `Task 1` backtory rewrites if needed.

2. **`_kickoff_sync` cancellation gap**: even with the `asyncio.to_thread` wrap, if `crew.kickoff` ignores cancellation and the thread pool is exhausted, `wait_for` may not actually interrupt execution. **Mitigation**: `FLOW_TIMEOUT_SECONDS=600` makes the timeout a "safety valve" rather than expected behavior; sub-2's actual runtimes for any ticker fit within 600s.

3. **LLM 4-tuple Literal extension**: `ValuationResult.method` Literal may need widening. **Mitigation**: Task 1 Step 9 is conditional — only widen if smoke test in Task 4 shows the LLM producing non-canonical values.

4. **byte-for-byte consistency break**: `InvestmentReport.rating` / `confidence` / `markdown` / `catalysts` are now LLM-derived and will differ per run. **Mitigation**: spec explicitly notes this; no test enforces byte-for-byte consistency on these fields.

5. **Graceful degradation regression**: if the company tool's `AllDataSourcesDown` re-raise logic is broken, ZZZZZZ will fall back to `INTERNAL_ERROR`. **Mitigation**: Task 4 has an explicit smoke test for this path.

6. **CrewAI 0.203.2 `output_pydantic` API drift**: the spec assumes `task_out.pydantic` attribute name. If CrewAI uses `task_out.pydantic_output` or stores it in `task_out.json_dict`, `_extract_pydantic_field` returns `None` and the field is marked unavailable — the flow continues with degraded output. **Mitigation**: implementer must verify the attribute name via Task 1's smoke run; if wrong, patch `_extract_pydantic_field` to use the actual name (no fallback per sub-3 strict-no-fallback decision).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-21-multi-agent-activation-sub3.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints
