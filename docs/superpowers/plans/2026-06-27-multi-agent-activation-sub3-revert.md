# Sub-Project 3 Revert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit the working-tree uncommitted changes for sub-3 revert (LLM drives 5 narrative fields via `ReportWriterOutput` slim Pydantic model; Flow computes 3 structured analyses deterministically) as 4 atomic commits + 1 validation pass.

**Architecture:** Most implementation already exists in the working tree (uncommitted). Each task is "review working-tree changes against spec, verify tests pass, fix any gaps, commit". Tasks map to the 4 commits from the spec §"实施顺序". The 5th task is real-LLM end-to-end validation against the spec §"验证清单".

**Tech Stack:** Python 3.11, CrewAI 0.203.2 (installed), LiteLLM, Pydantic v2, asyncio, pytest.

## Global Constraints

- Spec reference: `docs/superpowers/specs/2026-06-27-multi-agent-activation-sub3-revert-design.md` (read once, treat as authoritative)
- 4 commits, in order: 1) model, 2) crew+agents, 3) flow, 4) tests. Do NOT reorder.
- TDD is relaxed because tests already exist (commit 4 updates them). For each task, **verify existing tests pass** before committing.
- No new files except possibly `tests/conftest.py` (if a fixture isn't already extracted — check commit `eb77cdc` state)
- Do NOT amend previous commits. Create new commits even for small follow-ups.
- Do NOT push to remote
- Do NOT touch files outside the task's scope
- Each commit's subject line uses Conventional Commits (`feat(...)`, `test(...)`, `fix(...)`)
- Commit body references the spec path: `Sub-3 revert spec: docs/superpowers/specs/2026-06-27-multi-agent-activation-sub3-revert-design.md`

---

## Pre-flight (do once before Task 1)

- [ ] Verify clean test baseline

Run: `uv run pytest tests/ -q --tb=line`
Expected: `250 passed` (current working tree state). If failing, STOP and report — do not proceed.

- [ ] Confirm working-tree changes are uncommitted

Run: `git status --short`
Expected: 13 files modified (same list as shown in pre-plan review). If different, STOP and report.

- [ ] Read the spec once

Read: `docs/superpowers/specs/2026-06-27-multi-agent-activation-sub3-revert-design.md`
Purpose: internalize the architecture before touching code.

---

### Task 1: Commit `ReportWriterOutput` model

**Files:**
- Modify: `src/alphaquant/models/report.py` (verify has `ReportWriterOutput` + `_coerce_rating`)
- Modify: `src/alphaquant/models/__init__.py` (verify exports `ReportWriterOutput`)

**Interfaces:**
- Consumes: nothing (pure new model)
- Produces:
  - `class ReportWriterOutput(BaseModel)` in `alphaquant.models.report` with fields:
    - `rating: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]`
    - `confidence: int | None = Field(None, ge=0, le=100)`
    - `investment_horizon: Literal["short", "medium", "long"] = "medium"`
    - `catalysts: list[str] = Field(default_factory=list)`
    - `markdown: str = Field(..., min_length=1)`
    - `@field_validator("rating", mode="before")` named `_coerce_rating` that returns `"Hold"` if value not in allowed set
  - `ReportWriterOutput` exported from `alphaquant.models`

- [ ] **Step 1: Verify `ReportWriterOutput` exists in `src/alphaquant/models/report.py`**

Run: `grep -n "class ReportWriterOutput\|_coerce_rating" src/alphaquant/models/report.py`
Expected: 2 matches (the class definition and the validator).

If missing, STOP — do not proceed to Step 2. The model must exist before any commit.

- [ ] **Step 2: Verify `_coerce_rating` validator body matches spec**

Read: `src/alphaquant/models/report.py` lines around `_coerce_rating`
Expected:
```python
@field_validator("rating", mode="before")
@classmethod
def _coerce_rating(cls, v: Any) -> Any:
    allowed = {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}
    return v if v in allowed else "Hold"
```

If different, fix to match. (This is a "review and adjust" step, not "rewrite".)

- [ ] **Step 3: Verify `ReportWriterOutput` exported from `src/alphaquant/models/__init__.py`**

Run: `grep -n "ReportWriterOutput" src/alphaquant/models/__init__.py`
Expected: 1 match (the import or re-export).

If missing, add the export. Example:
```python
from alphaquant.models.report import InvestmentReport, ReportWriterOutput
```

- [ ] **Step 4: Run targeted tests**

Run: `uv run pytest tests/test_models_literals.py -q --tb=short`
Expected: all passed.

If any fail, fix the model or the test (whichever is wrong) and re-run.

- [ ] **Step 5: Run model-touching tests broadly**

Run: `uv run pytest tests/ -q -k "model or literal or report" --tb=short`
Expected: all passed.

- [ ] **Step 6: Stage and commit**

Run:
```bash
git add src/alphaquant/models/report.py src/alphaquant/models/__init__.py
git commit -m "$(cat <<'EOF'
feat(models): add ReportWriterOutput slim Pydantic for sub-3 revert

Sub-3 revert spec: docs/superpowers/specs/2026-06-27-multi-agent-activation-sub3-revert-design.md

ReportWriter agent now outputs a slim 5-field Pydantic model
(ReportWriterOutput) instead of the full InvestmentReport. MiniMax-M3
is reliable on slim models but emits structurally invalid output on
multi-field models with nested sub-scores (CompetitorAnalysis /
RiskAssessment / ValuationResult). The slim model covers LLM-driven
fields (rating / confidence / horizon / catalysts / markdown); the 3
structured analyses stay deterministic in the Flow.

_coerce_rating validator mirrors the established pattern from
CompetitorAnalysis._coerce_method and ValuationResult._coerce_method:
unknown rating values are coerced to "Hold" so the flow does not crash
on LLM mistakes.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

Expected: commit created. Verify with `git log -1 --stat` showing only the 2 staged files.

---

### Task 2: Commit crew + 4 analysis agent backtory changes

**Files:**
- Modify: `src/alphaquant/crews/analysis_crew.py`
- Modify: `src/alphaquant/agents/competitor_analyst.py`
- Modify: `src/alphaquant/agents/risk_analyst.py`
- Modify: `src/alphaquant/agents/valuation_analyst.py`
- Modify: `src/alphaquant/agents/report_writer.py`

**Interfaces:**
- Consumes: `ReportWriterOutput` from Task 1
- Produces:
  - `AnalysisCrew._TASK_TEMPLATES: list[tuple[str, str, type[BaseModel] | None]]` — 8 entries:
    - idx 0-3: `output_pydantic_model_or_None = None` (data tasks, sub-2 unchanged)
    - idx 4-6: `output_pydantic_model_or_None = None` (analysis tasks, REVERTED to text-only)
    - idx 7: `output_pydantic_model_or_None = ReportWriterOutput`
  - `AnalysisCrew._ASYNC_TASK_INDICES: set[int] = {0, 1, 2, 3, 4, 5, 6}` (report writer idx 7 is sequential)
  - `AnalysisCrew.tasks[7].context = [tasks[4], tasks[5], tasks[6]]` (ReportWriter depends on 3 analysis tasks)
  - 4 agent backtories reflect "text-only for analysis, slim Pydantic for report_writer"

- [ ] **Step 1: Verify `_TASK_TEMPLATES` shape and content**

Run: `grep -n "_TASK_TEMPLATES\|_ASYNC_TASK_INDICES\|_REPORT_WRITER_INDEX" src/alphaquant/crews/analysis_crew.py`
Expected: 3 distinct matches.

Read the `_TASK_TEMPLATES` definition. Verify:
- 8 entries
- Each is a 3-tuple `(role_key, description, pydantic_model_or_None)`
- Indices 4-6: pydantic_model = `None`
- Index 7: pydantic_model = `ReportWriterOutput`

If any deviation, fix in place (this is review-and-adjust).

- [ ] **Step 2: Verify `_ASYNC_TASK_INDICES = {0..6}`**

Run: `grep -A 2 "_ASYNC_TASK_INDICES" src/alphaquant/crews/analysis_crew.py | head -5`
Expected: `{0, 1, 2, 3, 4, 5, 6}` (or some equivalent set).

- [ ] **Step 3: Verify ReportWriter task has context**

Run: `grep -n "context=" src/alphaquant/crews/analysis_crew.py`
Expected: 1 match showing `context=[self.tasks[4], self.tasks[5], self.tasks[6]]` (or equivalent) for the report_writer task.

- [ ] **Step 4: Verify 3 analysis agent backtories say "text-only / Flow computes structured"**

For each of:
- `src/alphaquant/agents/competitor_analyst.py`
- `src/alphaquant/agents/risk_analyst.py`
- `src/alphaquant/agents/valuation_analyst.py`

Run: `grep -n "text-only\|Flow computes\|plain text\|summarize" <file>`
Expected: at least 1 match per file indicating the agent outputs text, not Pydantic.

If a backtory still implies Pydantic output, fix the backtory string to make it text-only.

- [ ] **Step 5: Verify ReportWriter backtory says "ReportWriterOutput 5 fields"**

Run: `grep -n "ReportWriterOutput\|output_pydantic\|slim" src/alphaquant/agents/report_writer.py`
Expected: at least 2 matches.

If ReportWriter backtory is missing the slim-model instruction, fix the backtory. (Note: as of pre-plan review, the ReportWriter backtory already includes the Confidence Rubric from commit `e8efef6` — that should remain.)

- [ ] **Step 6: Run crew tests**

Run: `uv run pytest tests/test_crew.py -q --tb=short`
Expected: all passed.

If any fail, diagnose:
- "tuple index out of range" → `_TASK_TEMPLATES` shape is wrong, fix Step 1
- "output_pydantic assertion failed" → Step 1 or 3 wrong, fix
- Otherwise: read failure message, locate responsible code, fix

- [ ] **Step 7: Run full test suite to catch any cross-cutting regressions**

Run: `uv run pytest tests/ -q --tb=line`
Expected: 250 passed (same baseline).

- [ ] **Step 8: Stage and commit**

Run:
```bash
git add src/alphaquant/crews/analysis_crew.py \
        src/alphaquant/agents/competitor_analyst.py \
        src/alphaquant/agents/risk_analyst.py \
        src/alphaquant/agents/valuation_analyst.py \
        src/alphaquant/agents/report_writer.py
git commit -m "$(cat <<'EOF'
feat(crew): revert 3 analysis tasks to text-only + slim report_writer Pydantic

Sub-3 revert spec: docs/superpowers/specs/2026-06-27-multi-agent-activation-sub3-revert-design.md

The original sub-3 plan (spec 2026-06-21-multi-agent-activation-sub3-design.md)
aimed for strict Pydantic output on all 4 analysis tasks. Real-LLM testing
revealed MiniMax-M3 emits structurally invalid output on multi-field models
(CompetitorAnalysis, RiskAssessment, ValuationResult) with nested sub-scores,
triggering CrewAI converter retry-loop until the 180s flow timeout.

Revert path: only ReportWriter (task idx 7) uses Pydantic output
(ReportWriterOutput slim 5-field model). The 3 analysis tasks
(competitor_analyst, risk_analyst, valuation_analyst) revert to text-only
output, used as context for ReportWriter via Task.context.

Backtories updated to reflect new responsibilities:
- 3 analysis agents: "summarize in plain text; Flow computes structured Pydantic"
- ReportWriter: "output ReportWriterOutput 5 fields: rating, confidence,
  horizon, catalysts, markdown" (Confidence Rubric preserved from e8efef6)

Async task indices widened from {0..3} to {0..6}: analysis text-only tasks
still parallelize for speed even though Flow does not read their output.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

Expected: commit created with 5 files modified.

---

### Task 3: Commit flow changes (deterministic helpers + run_crew + synthesize_report)

**Files:**
- Modify: `src/alphaquant/flows/analysis_flow.py`
- Modify: `src/alphaquant/interfaces/frontend/pages/1_Analyze.py`

**Interfaces:**
- Consumes: `ReportWriterOutput` from Task 1, crew from Task 2
- Produces (in `flows/analysis_flow.py`):
  - `FLOW_TIMEOUT_SECONDS = 300.0`
  - `_gics_peers_for(ticker, sector) -> list[Competitor]` — fallback peer set
  - `_compute_competitor_analysis(state) -> CompetitorAnalysis | None`
  - `_default_risk_subscores(state) -> list[RiskScore]`
  - `_compute_risk_assessment(state) -> RiskAssessment | None`
  - `_compute_valuation(state) -> ValuationResult | None`
  - `parse_crew_output(result, state)` — reads tasks 0-3 (data) and task 7 (writer_output); ignores tasks 4-6
  - `_extract_pydantic_field(tasks_output, idx, key, model_cls, state)` — used for ReportWriterOutput
  - `AnalysisState.writer_output: ReportWriterOutput | None = None`
  - `AnalysisFlow.run_crew` — uses `_kickoff_sync` (or equivalent inline sync helper) + `asyncio.to_thread` + `asyncio.wait_for(timeout=FLOW_TIMEOUT_SECONDS)`
  - `AnalysisFlow.synthesize_report` — 4 steps: compute 3 analyses → assemble `InvestmentReport` inline → fill runtime fields
- `interfaces/frontend/pages/1_Analyze.py` — adapts to new field structure (e.g., `writer_output` removed from report, `state.report` still works)

- [ ] **Step 1: Verify `FLOW_TIMEOUT_SECONDS = 300.0`**

Run: `grep -n "FLOW_TIMEOUT_SECONDS = " src/alphaquant/flows/analysis_flow.py`
Expected: `FLOW_TIMEOUT_SECONDS = 300.0` (single match).

If value differs (e.g., still 180), update it.

- [ ] **Step 2: Verify `AnalysisState.writer_output` field**

Run: `grep -n "writer_output" src/alphaquant/flows/analysis_flow.py`
Expected: multiple matches showing field declaration, parse_crew_output assignment, and synthesize_report use.

- [ ] **Step 3: Verify `parse_crew_output` only reads data fields + writer_output**

Run: `grep -n "_extract_data_field\|_extract_news_field\|_extract_pydantic_field" src/alphaquant/flows/analysis_flow.py | head -20`

Read the `parse_crew_output` function. Verify it ONLY reads:
- idx 0 (company_resolver)
- idx 1 (market_analyst)
- idx 2 (news_analyst)
- idx 3 (financial_analyst)
- idx 7 (report_writer → ReportWriterOutput)

It must NOT read idx 4/5/6 (those are text-only and ignored).

If `parse_crew_output` still tries to extract Pydantic from idx 4/5/6, remove those lines.

- [ ] **Step 4: Verify `_compute_competitor_analysis` / `_compute_risk_assessment` / `_compute_valuation` exist**

Run: `grep -n "^def _compute_competitor_analysis\|^def _compute_risk_assessment\|^def _compute_valuation\|^def _default_risk_subscores\|^def _gics_peers_for" src/alphaquant/flows/analysis_flow.py`
Expected: 5 matches (one per helper).

If missing, the helpers must exist. Read around line 200-350 to confirm body matches spec.

- [ ] **Step 5: Verify `run_crew` uses `asyncio.wait_for(asyncio.to_thread(...), timeout=FLOW_TIMEOUT_SECONDS)` pattern**

Run: `sed -n '540,590p' src/alphaquant/flows/analysis_flow.py`
Expected: code matching:
```python
def _kickoff_sync() -> CrewOutput:
    return AnalysisCrew().kickoff(inputs={"ticker": normalized})

result = await asyncio.wait_for(
    asyncio.to_thread(_kickoff_sync),
    timeout=FLOW_TIMEOUT_SECONDS,
)
```

If the pattern uses `asyncio.to_thread(crew.kickoff, inputs=...)` directly (without `_kickoff_sync` wrapper), refactor to wrap.

- [ ] **Step 6: Verify `synthesize_report` assembles `InvestmentReport` inline**

Run: `grep -n "self.state.report = InvestmentReport" src/alphaquant/flows/analysis_flow.py`
Expected: 1 match inside `synthesize_report`.

Read around the match. Verify it constructs the report with all required fields (company, market, financial, news, competitors, risk, valuation, rating, confidence, etc.).

- [ ] **Step 7: Verify frontend page adapts**

Run: `git diff HEAD -- src/alphaquant/interfaces/frontend/pages/1_Analyze.py | head -50`
Expected: a small diff adapting to new field structure (the diff should be small — fewer than 30 lines).

If the diff is large or shows unrelated changes, STOP and investigate before proceeding.

- [ ] **Step 8: Run flow tests**

Run: `uv run pytest tests/test_flow.py -q --tb=short`
Expected: all passed.

Common failures and fixes:
- `writer_output` assertion fails → Step 2 or 3 wrong
- timeout test asserts 180 but value is 300 → update assertion or revert timeout (decide based on spec: spec says 300, keep 300)
- "InvestmentReport missing field X" → Step 6 wrong, fix assembly

- [ ] **Step 9: Run full test suite**

Run: `uv run pytest tests/ -q --tb=line`
Expected: 250 passed (same baseline).

- [ ] **Step 10: Stage and commit**

Run:
```bash
git add src/alphaquant/flows/analysis_flow.py \
        src/alphaquant/interfaces/frontend/pages/1_Analyze.py
git commit -m "$(cat <<'EOF'
feat(flow): compute competitor/risk/valuation deterministically + assemble report from writer_output

Sub-3 revert spec: docs/superpowers/specs/2026-06-27-multi-agent-activation-sub3-revert-design.md

The 3 sub-2 deferred blockers are fixed in this commit:

1. Asyncio shutdown race: run_crew wraps crew.kickoff in _kickoff_sync +
   asyncio.to_thread, allowing asyncio.wait_for to cancel mid-execution.
   parse_crew_output remains pure sync.

2. FLOW_TIMEOUT_SECONDS: 180 -> 300. 7 LLM tasks with average 20-40s each
   plus CrewAI manager/converter overhead = 50-100s; 300s leaves 1.5-2x
   buffer. Original spec's 600s was overly conservative.

3. Tool empty-shell fallback: removed in commit 8d1412e (already in main).
   4 data tools now return error strings on failure instead of
   {"name": "N/A", ...} empty shells.

Plus the sub-3 revert: 3 analysis fields (competitor / risk / valuation)
are now computed deterministically via inline _compute_* helpers in
flows/analysis_flow.py (not extracted to scoring/{competitive,risk_score}
modules as originally planned). synthesize_report assembles InvestmentReport
inline from data + 3 analyses + writer_output.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

Expected: commit created with 2 files modified.

---

### Task 4: Commit test updates

**Files:**
- Modify: `tests/test_flow.py`
- Modify: `tests/test_crew.py`
- Modify: `tests/test_models_literals.py`
- Possibly modify: `tests/conftest.py` (only if fixture dedup needed)

**Interfaces:**
- Consumes: implementations from Tasks 1-3
- Produces: tests that exercise sub-3 revert behavior

- [ ] **Step 1: Run full test suite to confirm current state**

Run: `uv run pytest tests/ -q --tb=line`
Expected: 250 passed.

If failing, STOP and report — there should be no test failures after Tasks 1-3.

- [ ] **Step 2: Verify `tests/conftest.py` has shared `InvestmentReport` fixture (if needed)**

Run: `grep -n "def investment_report\|def sample_report" tests/conftest.py`
Expected: at least 1 match (commit `eb77cdc` already extracted this fixture).

If missing AND any test file defines `InvestmentReport(...)` inline, extract to conftest. Otherwise skip.

- [ ] **Step 3: Verify `tests/test_flow.py` covers sub-3 revert specifics**

Run: `grep -n "writer_output\|FLOW_TIMEOUT_SECONDS\|_compute_competitor\|_compute_risk\|_compute_valuation" tests/test_flow.py`
Expected: at least 4 matches covering writer_output extraction, timeout 300, and at least one of the compute_* helpers.

If a critical sub-3 behavior is missing a test, add it. Examples:
- `test_run_crew_uses_kickoff_sync_with_wait_for`: assert run_crew wraps kickoff in sync helper
- `test_synthesize_report_assembles_investment_report_inline`: assert report fields come from state
- `test_writer_output_none_appends_error`: assert graceful failure
- `test_flow_timeout_seconds_is_300`: constant check

- [ ] **Step 4: Verify `tests/test_crew.py` covers sub-3 revert specifics**

Run: `grep -n "ReportWriterOutput\|output_pydantic\|_ASYNC_TASK_INDICES" tests/test_crew.py`
Expected: at least 4 matches.

- [ ] **Step 5: Verify `tests/test_models_literals.py` covers Literal widening**

Run: `grep -n "blended\|dcf_only\|hybrid\|multi_factor" tests/test_models_literals.py`
Expected: at least 4 matches.

- [ ] **Step 6: Run full test suite one more time**

Run: `uv run pytest tests/ -q --tb=line`
Expected: 250 passed (or more if Step 3 added tests).

- [ ] **Step 7: Stage and commit**

Run:
```bash
git add tests/test_flow.py tests/test_crew.py tests/test_models_literals.py tests/conftest.py 2>/dev/null || \
git add tests/test_flow.py tests/test_crew.py tests/test_models_literals.py
git commit -m "$(cat <<'EOF'
test(sub-3): cover ReportWriterOutput extraction, timeout=300, literal widening

Sub-3 revert spec: docs/superpowers/specs/2026-06-27-multi-agent-activation-sub3-revert-design.md

Tests aligned with the sub-3 revert implementation:

tests/test_flow.py:
- TestParseCrewOutput covers writer_output extraction (idx 7 Pydantic)
  and confirms tasks 4-6 (analysis text-only) are NOT extracted
- TestRunCrewStep asserts run_crew wraps kickoff in _kickoff_sync +
  asyncio.to_thread + asyncio.wait_for pattern (sub-2 Blocker 1 fix)
- timeout assertion updated from 180 to 300 (sub-3 spec)

tests/test_crew.py:
- test_task_templates_uses_3_tuple_with_pydantic_model: idx 4-6 have
  pydantic_model=None, idx 7 has pydantic_model=ReportWriterOutput
- test_async_task_indices_cover_data_and_analysis_not_report: {0..6}
- test_report_writer_task_has_context_with_analysis_tasks: context=[4,5,6]
- 3 tests confirming analysis tasks have NO output_pydantic
- 1 test confirming report_writer task HAS output_pydantic

tests/test_models_literals.py:
- ValuationResult.method: all 6 widened values accepted
- CompetitorAnalysis.method: all 7 widened values accepted
- _coerce_method fallback to "dcf_relative_peg" / "gics" for unknown values
- Verbose LLM description coerced to default (LLM guard pattern)

tests/conftest.py: InvestmentReport fixture already extracted in
commit eb77cdc; no change needed.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

Note: The `2>/dev/null || ...` fallback handles the case where `tests/conftest.py` doesn't exist or has no changes to stage.

Expected: commit created.

---

### Task 5: Real LLM end-to-end validation

**Files:** None modified. This task is validation only.

**Goal:** Validate the spec §"验证清单" against real LLM behavior.

**Pre-conditions:**
- Tasks 1-4 complete and committed
- `.env` has `MINIMAX_API_KEY` set (real key per `MEMORY.md`)
- `LITELLM_MODEL=openai/MiniMax-M3` (NOT `minimax/minimax-m3` per `MEMORY.md`)

- [ ] **Step 1: Verify environment**

Run: `grep "LITELLM_MODEL" .env`
Expected: `LITELLM_MODEL=openai/MiniMax-M3`

If different, STOP — fix `.env` first.

Run: `grep "MINIMAX_API_KEY" .env | head -c 30`
Expected: line starts with `MINIMAX_API_KEY=` and has value (placeholder would be `your-key-here` or similar).

If placeholder, STOP — set real key first.

- [ ] **Step 2: Run AAPL end-to-end**

Run: `time python -m alphaquant AAPL --format json > /tmp/aapl.json 2> /tmp/aapl.log`
Expected: completes in <300 seconds.

Verify the output is valid JSON and contains required fields:
```bash
python -c "
import json
with open('/tmp/aapl.json') as f:
    r = json.load(f)
assert r['rating'] in {'Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'}, f'bad rating: {r[\"rating\"]}'
assert r['confidence'] is None or (0 <= r['confidence'] <= 100), f'bad confidence: {r[\"confidence\"]}'
assert len(r['markdown']) > 100, 'markdown too short'
print(f'AAPL rating={r[\"rating\"]} confidence={r[\"confidence\"]} markdown_len={len(r[\"markdown\"])}')
"
```

If assertion fails, save `/tmp/aapl.log` and STOP — report failure mode.

- [ ] **Step 3: Run MSFT end-to-end**

Run: `time python -m alphaquant MSFT --format json > /tmp/msft.json 2> /tmp/msft.log`
Expected: completes in <300 seconds.

Same validation as Step 2.

- [ ] **Step 4: Run TSLA end-to-end**

Run: `time python -m alphaquant TSLA --format json > /tmp/tsla.json 2> /tmp/tsla.log`
Expected: completes in <300 seconds.

Same validation as Step 2.

- [ ] **Step 5: Verify 3 tickers produce DIFFERENT LLM-driven fields**

Run:
```bash
python -c "
import json
files = ['/tmp/aapl.json', '/tmp/msft.json', '/tmp/tsla.json']
data = [json.load(open(f)) for f in files]
confs = [d['confidence'] for d in data]
ratings = [d['rating'] for d in data]
markdowns = [d['markdown'] for d in data]
assert len(set(confs)) > 1, f'all 3 confidence identical: {confs}'
assert len(set(ratings)) > 1 or len(set(markdowns)) > 1, f'LLM-driven fields identical across tickers'
print(f'confs={confs}, ratings={ratings}')
"
```

Expected: 3 confidence values are NOT all the same number (LLM is genuinely deciding).

If all 3 confidence are identical, the LLM is using a fixed formula — STOP and report.

- [ ] **Step 6: Run ZZZZZZ (failure case)**

Run: `time python -m alphaquant ZZZZZZ 2> /tmp/zzzzzz.log; echo "exit=$?"`
Expected: exits with non-zero status, log shows `AllDataSourcesDown`, total time <30 seconds.

If exit is 0 OR time is >30 seconds, the graceful degradation is broken — STOP and report.

- [ ] **Step 7: Final code-search verification (per spec §"代码搜索")**

Run: `grep -r "scoring.rating" src/alphaquant/flows/`
Expected: 0 matches.

Run: `grep -r "deterministic_fallback" src/alphaquant/flows/`
Expected: 0 matches.

Run: `grep "FLOW_TIMEOUT_SECONDS = 300" src/alphaquant/flows/analysis_flow.py`
Expected: 1 match.

Run: `grep "_coerce_rating" src/alphaquant/models/report.py`
Expected: 1 match.

If any grep returns unexpected results, fix in a follow-up commit (do not amend Task 1-4).

- [ ] **Step 8: Validation report**

Write a short report summarizing:
- AAPL/MSFT/TSLA actual time elapsed, rating, confidence, markdown length
- ZZZZZZ failure mode (expected: AllDataSourcesDown)
- 3 confidence values (proves LLM is determining, not formula)
- Any spec checklist items that failed

If all pass: sub-3 revert is COMPLETE. Inform the user.

If any fail: list failures and STOP — do not declare completion.

---

## Self-Review Notes (do not skip)

After Task 5, the spec is fully implemented and validated. Sub-3 revert is complete.

Future work (sub-4, separate spec): `allow_delegation=True`, CrewAI Memory, retry strategy, progressive degrade.
