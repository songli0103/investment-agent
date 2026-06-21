# Confidence Rubric + Optional Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `report.confidence` from defaulting to 67 (LLM behavior) by giving the report_writer LLM an explicit rubric + reasoning section, while letting the LLM return `null` when it cannot justify a number — and propagating `None` safely through DB and frontend.

**Architecture:** 4 small changes — Pydantic field widens to `int | None`, prompt gains a rubric, DB schema migrates column to nullable, frontend/Streamlit guards `None`. No new dependencies. No business-logic rewrites.

**Tech Stack:** Pydantic v2 (`Field(default=None, ge=0, le=100)`), SQLite (ALTER TABLE migration), Streamlit (display guards), CrewAI 0.203.2 (agent backstory string).

## Global Constraints

- **Branch:** `main` (single-developer convention; commit + push when done).
- **Test baseline:** 239/239 passing on commit `5553e43` (sub-3 ship point). Plan must end at ≥239.
- **Commits:** Frequent, conventional (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), Co-authored-by Claude Opus 4.6.
- **Spec authority:** `docs/superpowers/specs/2026-06-21-confidence-rubric-design.md` (committed `b0308df`). If a step conflicts with the spec, the spec governs — raise to user.
- **DB migration:** must be additive (no `DROP TABLE`); use SQLite `ALTER TABLE ... DROP NOT NULL` if SQLite version supports it, else create new table + copy.
- **Scope expansion vs spec:** Plan adds DB-layer changes (Task 3) and `tests/smoke.py` guard (Task 5) that the spec marked "out of scope" but which are required for the optional contract to not crash on insert. Disclosed here; not a silent addition.

---

## Task 1: Make `InvestmentReport.confidence` optional + add unit tests

**Files:**
- Modify: `src/alphaquant/models/report.py:33`
- Create: `tests/test_report_optional.py`

**Interfaces:**
- Consumes: existing `InvestmentReport` model (no other changes needed).
- Produces: `InvestmentReport.confidence: int | None = Field(None, ge=0, le=100)` — accepting both numbers and `None`.

- [ ] **Step 1: Write the failing test file `tests/test_report_optional.py`**

```python
"""Tests for optional InvestmentReport.confidence field.

Sub-plan for confidence-rubric spec (b0308df). confidence becomes int | None
so the LLM can return null when it cannot justify a number.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from alphaquant.models.report import InvestmentReport


def _stub_report(**overrides: Any) -> InvestmentReport:
    """Build a minimal valid InvestmentReport, allowing field overrides.

    Fields other than `confidence` use known-good values; the test only cares
    about confidence validation. Caller can override any field via kwargs.
    """
    base: dict[str, Any] = {
        "report_id": "00000000-0000-0000-0000-000000000000",
        "ticker": "AAPL",
        "generated_at": datetime(2024, 1, 1, 0, 0, 0),
        "data_as_of": {},
        # company/market/financial/news/competitors/risk/valuation — use None
        # to bypass required-model construction; confidence is what we test.
        "company": None,
        "market": None,
        "financial": None,
        "financial_health_score": 70,
        "news": None,
        "competitors": None,
        "risk": None,
        "valuation": None,
        "rating": "Hold",
        "investment_horizon": "medium",
        "catalysts": [],
        "markdown": "## Summary",
        "sources": [],
        "disclaimer": "test",
    }
    base.update(overrides)
    return InvestmentReport(**base)


class TestConfidenceOptional:
    def test_none_accepted(self):
        """Confidence can be None (LLM returns null when not justified)."""
        rep = _stub_report(confidence=None)
        assert rep.confidence is None

    def test_default_is_none(self):
        """When confidence is omitted, default is None (not ValidationError)."""
        rep = _stub_report()
        assert rep.confidence is None

    def test_zero_accepted(self):
        rep = _stub_report(confidence=0)
        assert rep.confidence == 0

    def test_hundred_accepted(self):
        rep = _stub_report(confidence=100)
        assert rep.confidence == 100

    def test_seventy_accepted(self):
        """Regression: numeric confidence values still work."""
        rep = _stub_report(confidence=70)
        assert rep.confidence == 70

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            _stub_report(confidence=-1)

    def test_above_100_rejected(self):
        with pytest.raises(ValidationError):
            _stub_report(confidence=101)
```

Note: if `_stub_report` cannot construct due to `company`/`market`/etc. being required, the implementer should adapt by supplying the minimal valid nested models instead of `None`. Verify by running the test.

- [ ] **Step 2: Run the new tests to verify they fail (None + default cases should fail)**

Run: `uv run pytest tests/test_report_optional.py -v`
Expected: `test_none_accepted` and `test_default_is_none` FAIL with `ValidationError: confidence Field required` (because field is currently `int = Field(..., ...)`). Other tests may also fail if nested stubs are not minimal. **Document the actual failures in the task report.**

- [ ] **Step 3: Make `InvestmentReport.confidence` optional**

In `src/alphaquant/models/report.py:33`, change:

```python
confidence: int = Field(..., ge=0, le=100)
```

to:

```python
confidence: int | None = Field(None, ge=0, le=100)
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_report_optional.py -v`
Expected: All 7 tests in `TestConfidenceOptional` pass.

- [ ] **Step 5: Run full test suite to verify no regression**

Run: `uv run pytest tests/ -q`
Expected: 239 + 7 new = **246 passed**, 0 failed. If existing tests fail, investigate (likely a test that didn't pass `confidence=` explicitly — should not exist per spec, but verify).

- [ ] **Step 6: Commit**

```bash
git add src/alphaquant/models/report.py tests/test_report_optional.py
git commit -m "$(cat <<'EOF'
feat(models): make InvestmentReport.confidence optional

LLM-driven confidence can now return null when it cannot justify a number.
Adds tests/test_report_optional.py with 7 cases pinning the new contract.

Confidence still constrained to ge=0, le=100 when provided.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add explicit Confidence Rubric + Markdown Rationale to `report_writer` prompt

**Files:**
- Modify: `src/alphaquant/agents/report_writer.py` (the `backstory` string)

**Interfaces:**
- Consumes: existing `report_writer` agent construction (no signature change).
- Produces: same `Agent` instance; the LLM receives an explicit rubric + markdown-section requirement in its backstory. No schema change; pure prompt change.

- [ ] **Step 1: Read current backstory**

Re-read `src/alphaquant/agents/report_writer.py` (lines 15–27) to confirm exact text before editing. The backstory ends with:

```
"rating and confidence must reflect the actual risk and valuation signals, not "
"a fixed formula."
```

- [ ] **Step 2: Replace the backstory's confidence-related sentence with rubric**

In `src/alphaquant/agents/report_writer.py`, replace the final sentence:

```python
        "rating and confidence must reflect the actual risk and valuation signals, not "
        "a fixed formula.",
```

with the rubric block (note: keep the `"rating must reflect..."` phrasing intact for `rating`; only the `confidence` sentence is replaced):

```python
        "rating must reflect the actual risk and valuation signals, not "
        "a fixed formula. "
        "confidence uses this rubric — pick a band, then defend it in markdown:\n"
        "  - 80-100: Strong conviction. 5/5 data sources present (company, market, "
        "financial, news, competitor); DCF and relative valuation agree within 20%; "
        "risk level low or medium; news sentiment not extreme.\n"
        "  - 60-79: Moderate conviction. 4/5 data sources; DCF/relative agree within "
        "40%; risk low or medium; OR one weak signal with no major contradictions.\n"
        "  - 40-59: Low conviction. 3/5 data sources; OR DCF/relative diverge >40%; "
        "OR risk high; OR news sentiment extreme.\n"
        "  - 20-39: Weak conviction. ≤2 data sources; OR risk extreme; OR major "
        "contradictions among signals.\n"
        "  - 0-19 or null: Cannot evaluate. Set confidence=null and document why in "
        "markdown. If unsure, null is safer than guessing a number.\n"
        "Markdown MUST include a '## Confidence Rationale' section listing: "
        "data sources present (e.g. '5/5: company, market, financial, news, "
        "competitor'); DCF vs relative agreement (e.g. 'DCF $180 vs relative $175, "
        "3% spread'); risk level (low/medium/high/extreme); any extreme signals; "
        "one-sentence verdict explaining why this confidence number (or null) was "
        "chosen.",
```

- [ ] **Step 3: Verify the file parses and the prompt is wired**

Run: `uv run python -c "from alphaquant.agents.report_writer import build_report_writer_agent; from crewai.llm import LLM; a = build_report_writer_agent(LLM(model='openai/MiniMax-M3')); print('OK'); print('---'); print(a.backstory[:500])"`
Expected: `OK` then the rubric text begins with `confidence uses this rubric`. If the prompt string is corrupted, the import or construction will fail.

- [ ] **Step 4: Run full test suite — no behavioral test should break (pure prompt change)**

Run: `uv run pytest tests/ -q`
Expected: 246 passed. No regression (prompt change has no unit-test coverage — that's expected; verification of LLM behavior awaits B7/B8 fix).

- [ ] **Step 5: Commit**

```bash
git add src/alphaquant/agents/report_writer.py
git commit -m "$(cat <<'EOF'
feat(agent): add explicit Confidence Rubric + markdown rationale to report_writer

The vague 'reflect actual signals, not a fixed formula' phrasing let the LLM
default to mid-range (observed: 67). New backstory block enumerates 5 bands
with concrete criteria (data source count, DCF/relative agreement, risk
level, news sentiment). Markdown must contain '## Confidence Rationale'
section listing the inputs and a one-sentence verdict. 0-19 / null is the
explicit 'cannot evaluate' band.

No schema change. Prompt-only — real-LLM verification awaits B7/B8 fix.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: DB layer + ReportRecord accept `None` confidence (with migration)

**Files:**
- Modify: `src/alphaquant/infrastructure/persistence/db.py`
- Modify: `src/alphaquant/infrastructure/persistence/models.py`
- Modify: `tests/test_db.py` (add nullable-confidence round-trip test)

**Interfaces:**
- Consumes: `InvestmentReport` with `confidence: int | None` (from Task 1).
- Produces: `ReportRecord.confidence: int | None`; DB column allows NULL; `insert_report` and `_row_to_record` tolerate None.

- [ ] **Step 1: Update `ReportRecord.confidence` to optional**

In `src/alphaquant/infrastructure/persistence/models.py:16`, change:

```python
confidence: int
```

to:

```python
confidence: int | None
```

- [ ] **Step 2: Update SQLite schema to make `confidence` nullable**

In `src/alphaquant/infrastructure/persistence/db.py:14-26`, the `SCHEMA` constant uses `CREATE TABLE IF NOT EXISTS`. Existing DBs won't get the column change automatically. Add a migration helper.

Replace the `SCHEMA` constant with:

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    rating TEXT NOT NULL,
    confidence INTEGER,
    market_price REAL,
    report_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_ticker ON reports(ticker);
CREATE INDEX IF NOT EXISTS idx_reports_generated_at ON reports(generated_at);
"""

# Migration for DBs created before confidence became nullable. SQLite ≥3.35
# supports ALTER TABLE ... ALTER COLUMN ... DROP NOT NULL; fall back to
# table-rebuild for older versions.
_MIGRATION_V2 = "ALTER TABLE reports ALTER COLUMN confidence DROP NOT NULL;"


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Run additive migrations idempotently."""
    cur = conn.execute("PRAGMA user_version")
    version = int(cur.fetchone()[0])
    if version < 2:
        try:
            conn.execute(_MIGRATION_V2)
            conn.execute("PRAGMA user_version = 2")
        except sqlite3.OperationalError:
            # Older SQLite: table-rebuild fallback.
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS reports_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    confidence INTEGER,
                    market_price REAL,
                    report_json TEXT NOT NULL
                );
                INSERT INTO reports_new
                    SELECT id, ticker, generated_at, rating, confidence,
                           market_price, report_json FROM reports;
                DROP TABLE reports;
                ALTER TABLE reports_new RENAME TO reports;
                CREATE INDEX IF NOT EXISTS idx_reports_ticker
                    ON reports(ticker);
                CREATE INDEX IF NOT EXISTS idx_reports_generated_at
                    ON reports(generated_at);
                PRAGMA user_version = 2;
                """
            )
        conn.commit()
```

Note: `PRAGMA user_version` is the standard SQLite migration pattern in this DB. **Verify `PRAGMA user_version` is currently 1 (or some value <2) before assuming migration needed.** If the existing DB never used `PRAGMA user_version`, this PRAGMA call will simply return 0 and the migration will run unconditionally. That is acceptable.

Also update `init()` to run migrations after schema creation:

Replace `init()` body with:

```python
def init(self) -> None:
    """Create reports table and indexes if absent, then run migrations."""
    with self._connect() as conn:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
        conn.commit()
```

- [ ] **Step 3: Update `_row_to_record` to tolerate None confidence**

In `src/alphaquant/flows/...db.py:128-140`, replace:

```python
confidence=int(row["confidence"]),
```

with:

```python
confidence=(
    int(row["confidence"]) if row["confidence"] is not None else None
),
```

The `insert_report` method (line 63) already passes `report.confidence` directly — when confidence is `None`, SQLite will store NULL (now that the column is nullable). No code change needed in `insert_report`.

- [ ] **Step 4: Write a failing test for nullable round-trip**

Append to `tests/test_db.py`:

```python
class TestNullableConfidence:
    def test_insert_and_read_with_null_confidence(self, tmp_path):
        """Sub-plan: confidence can be null in InvestmentReport → DB → row."""
        from alphaquant.infrastructure.persistence.db import DB

        db_path = tmp_path / "null_conf.db"
        db = DB(db_path)
        db.init()
        # Build a minimal report with confidence=None
        from datetime import datetime
        from alphaquant.models.report import InvestmentReport
        rep = InvestmentReport(
            report_id="00000000-0000-0000-0000-000000000001",
            ticker="TEST",
            generated_at=datetime(2024, 1, 1),
            data_as_of={},
            company=None,
            market=None,
            financial=None,
            financial_health_score=50,
            news=None,
            competitors=None,
            risk=None,
            valuation=None,
            rating="Hold",
            confidence=None,  # <-- the change under test
            investment_horizon="medium",
            catalysts=[],
            markdown="x",
            sources=[],
            disclaimer="x",
        )
        new_id = db.insert_report("TEST", rep)
        rows = db.get_history()
        assert len(rows) == 1
        assert rows[0].id == new_id
        assert rows[0].confidence is None

    def test_insert_and_read_with_numeric_confidence(self, tmp_path):
        """Regression: numeric confidence still round-trips."""
        from alphaquant.infrastructure.persistence.db import DB
        from datetime import datetime
        from alphaquant.models.report import InvestmentReport

        db_path = tmp_path / "num_conf.db"
        db = DB(db_path)
        db.init()
        rep = InvestmentReport(
            report_id="00000000-0000-0000-0000-000000000002",
            ticker="TEST",
            generated_at=datetime(2024, 1, 1),
            data_as_of={},
            company=None,
            market=None,
            financial=None,
            financial_health_score=50,
            news=None,
            competitors=None,
            risk=None,
            valuation=None,
            rating="Hold",
            confidence=80,
            investment_horizon="medium",
            catalysts=[],
            markdown="x",
            sources=[],
            disclaimer="x",
        )
        db.insert_report("TEST", rep)
        rows = db.get_history()
        assert rows[0].confidence == 80
```

Verify `tests/test_db.py` already has a `tmp_path` fixture or uses `conftest.py`. If not, check `tests/conftest.py` (per sub-3 final review, that file exists). Adjust the test to use the existing fixture pattern.

- [ ] **Step 5: Run new tests + full DB test module**

Run: `uv run pytest tests/test_db.py -v`
Expected: New `TestNullableConfidence` tests pass; existing DB tests still pass.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: 246 + 2 new = **248 passed**.

- [ ] **Step 7: Commit**

```bash
git add src/alphaquant/infrastructure/persistence/db.py src/alphaquant/infrastructure/persistence/models.py tests/test_db.py
git commit -m "$(cat <<'EOF'
fix(db): allow NULL confidence + migration for existing DBs

InvestmentReport.confidence became optional in the previous commit; the DB
layer previously enforced NOT NULL on the confidence column and stored it
as int (not int | None). Without this commit, reports with confidence=null
would crash on insert and on read.

Changes:
- SCHEMA: confidence INTEGER (was INTEGER NOT NULL)
- PRAGMA user_version migration v1→v2 with ALTER COLUMN fallback to
  table-rebuild for SQLite < 3.35
- ReportRecord.confidence: int | None
- _row_to_record: tolerate None on read

Round-trip test pins the new contract (numeric + null both round-trip).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Frontend consumers tolerate `None` confidence

**Files:**
- Modify: `src/alphaquant/interfaces/frontend/components/rating_card.py:32-55`
- Modify: `src/alphaquant/interfaces/frontend/pages/2_History.py:83`
- Modify: `src/alphaquant/interfaces/frontend/pages/3_Compare.py:91, 232-234`

**Interfaces:**
- Consumes: `InvestmentReport.confidence: int | None` and `ReportRecord.confidence: int | None`.
- Produces: Streamlit renders `"N/A"` or skips rather than `f"{None}%"` / TypeError in `max()`.

- [ ] **Step 1: Update `rating_card.py` to render `"N/A"` when confidence is None**

In `src/alphaquant/interfaces/frontend/components/rating_card.py:32`, change:

```python
confidence = report.confidence
```

to:

```python
confidence = report.confidence if report.confidence is not None else "N/A"
```

Then in the f-string (line 55), change `{confidence}%` to `{confidence}{'%' if isinstance(confidence, int) else ''}` so that `100` renders as `100%` and `None` (now `"N/A"`) renders as `N/A` (no `%`).

Final rendering logic:

```python
confidence = report.confidence if report.confidence is not None else "N/A"
...
<div style="font-size: 2.0rem; font-weight: 600; line-height: 1.1; margin-top: 4px;">
    {confidence}{"%" if isinstance(confidence, int) else ""}
</div>
```

- [ ] **Step 2: Update `pages/2_History.py:83` to guard None**

Read the file to find the exact context. The relevant line:

```python
confidence=int(row["confidence"]),
```

Replace with:

```python
confidence=(int(row["confidence"]) if row["confidence"] is not None else None),
```

Also check line 40 (`"confidence": r.confidence` in a dict) — if `r.confidence` is `None`, that's fine for dict; no change needed unless the consumer page later does arithmetic.

- [ ] **Step 3: Update `pages/3_Compare.py` to skip None in `max()`**

Read the file to find lines 229-238. The relevant logic:

```python
best_confidence_ticker = max(successful, key=lambda r: r.confidence).ticker
```

This will crash if any `r.confidence` is `None` (TypeError: '>' not supported between 'int' and 'NoneType'). Replace with a sort that treats `None` as `-1` (lower than any real confidence):

```python
best_confidence_ticker = max(
    successful,
    key=lambda r: (r.confidence if r.confidence is not None else -1),
).ticker
```

Also check line 232 (`key=lambda r: (RATING_TO_NUMERIC.get(r.rating, 0), r.confidence)`) — if any `r.confidence` is `None`, this tuple-comparison will TypeError. Replace with:

```python
key=lambda r: (
    RATING_TO_NUMERIC.get(r.rating, 0),
    r.confidence if r.confidence is not None else -1,
),
```

If `successful` is empty (no successful reports), the original code would also crash; leave that as-is since it was already an edge case.

- [ ] **Step 4: Write a small unit test for `rating_card.render_rating_card` if practical**

Streamlit components are usually hard to unit-test (require `streamlit` runtime). Check if existing tests exist for `rating_card.py`:

Run: `grep -r "render_rating_card" tests/` (or similar).

If a unit test exists for it, add:

```python
def test_render_rating_card_handles_none_confidence():
    """confidence=None should render without TypeError."""
    from alphaquant.interfaces.frontend.components.rating_card import (
        render_rating_card,
    )
    # Build a minimal InvestmentReport with confidence=None
    rep = _stub_report(confidence=None)
    # Streamlit's markdown call is what would crash; if testing with
    # @streamlit.testing.v1.AppTest, use that. Otherwise skip and rely
    # on manual smoke.
    # ...
```

If no clean way to unit-test, **skip this step** and add a manual smoke step to Task 5. Document the choice in the task report.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: 248 passed (frontend changes are not unit-tested; verified via manual smoke in Task 5).

- [ ] **Step 6: Manual smoke check (optional but recommended)**

If the environment can launch Streamlit, run:

```bash
uv run streamlit run src/alphaquant/interfaces/frontend/app.py
```

Navigate to the Analyze page; if there are existing reports with `confidence=None`, the rating card should show `N/A`. If there are no such reports, the manual check is moot. Skip this step if Streamlit cannot launch in the environment.

- [ ] **Step 7: Commit**

```bash
git add src/alphaquant/interfaces/frontend/components/rating_card.py src/alphaquant/interfaces/frontend/pages/2_History.py src/alphaquant/interfaces/frontend/pages/3_Compare.py
git commit -m "$(cat <<'EOF'
fix(frontend): guard confidence=None in rating_card, Compare, History

Confidence became optional in the prior commit. Three frontend consumers
would crash on None:

- rating_card.py rendered 'None%' (ugly string interpolation)
- 2_History.py did int(None) on read
- 3_Compare.py used None in max() tuple comparison (TypeError)

Fix: rating_card shows 'N/A' when None; History/Compare coerce None to
-1 for ordering purposes (treats unknown as lowest).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update `tests/smoke.py` + final test sweep

**Files:**
- Modify: `tests/smoke.py:72`

**Interfaces:**
- Consumes: `InvestmentReport` with `confidence: int | None`.
- Produces: smoke test that tolerates None without TypeError.

- [ ] **Step 1: Update the smoke test assertion**

In `tests/smoke.py:72`, change:

```python
assert_(0 <= report.confidence <= 100, f"Confidence {report.confidence} in [0,100]")
```

to:

```python
if report.confidence is not None:
    assert_(0 <= report.confidence <= 100, f"Confidence {report.confidence} in [0,100]")
```

(Use existing `assert_` helper if present; otherwise use plain `assert`. Read the file's import block first to confirm.)

- [ ] **Step 2: Run full test suite for final sweep**

Run: `uv run pytest tests/ -q`
Expected: **248 passed, 0 failed** (246 baseline + 2 new in TestNullableConfidence).

- [ ] **Step 3: Quick lint/type sanity (optional)**

Run: `uv run python -c "from alphaquant.models.report import InvestmentReport; print(InvestmentReport(confidence=None).confidence)"`
Expected: `None`

Run: `uv run python -c "from alphaquant.agents.report_writer import build_report_writer_agent; from crewai.llm import LLM; a = build_report_writer_agent(LLM(model='openai/MiniMax-M3')); assert 'Confidence Rubric' in a.backstory or 'confidence uses this rubric' in a.backstory; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add tests/smoke.py
git commit -m "$(cat <<'EOF'
test(smoke): tolerate confidence=None in range assertion

Confidence is now optional; smoke should not TypeError on None.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification (after all 5 tasks complete)

- [ ] `uv run pytest tests/ -q` → **248 passed**, 0 failed
- [ ] `git log --oneline -8` shows 5 new commits on `main` ahead of `5553e43`:
  - `feat(models): make InvestmentReport.confidence optional` (+ 7 tests)
  - `feat(agent): add explicit Confidence Rubric ...`
  - `fix(db): allow NULL confidence + migration for existing DBs`
  - `fix(frontend): guard confidence=None ...`
  - `test(smoke): tolerate confidence=None ...`
- [ ] No unintended files modified (only the 6 listed in "Files Touched" across all tasks)
- [ ] Spec checklist from `2026-06-21-confidence-rubric-design.md` satisfied:
  - [x] `InvestmentReport(confidence=None)` accepted (Task 1)
  - [x] `confidence=0/50/100` accepted; `confidence=-1/101` rejected (Task 1)
  - [x] `report_writer.py` backstory contains rubric (Task 2)
  - [x] All 239 baseline tests still pass; +9 new tests total
  - [x] Streamlit renders `None` without crash (Task 4 — guarded at all 3 sites)
  - [x] DB schema migrated; round-trip verified (Task 3)

## Out of Scope (Explicit)

- Real-LLM end-to-end verification (B7/B8 still pending; 180s timeout still SIGTERM).
- DCF wiring into the flow (`scoring/dcf.py::compute_dcf_value` remains dormant; the valuation_analyst agent has its own `DCFTool`).
- Frontend redesign of how `None` confidence is visualized (currently shows `N/A`; could be a "Not Rated" badge in a future iteration).