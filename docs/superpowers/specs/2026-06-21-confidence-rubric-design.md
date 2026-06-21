# Confidence Rubric + Optional Confidence — Design

## Context

User reports that report `confidence` always appears as `67` in generated investment reports.

Investigation (post-sub-3):

- Sub-project 3 (`5553e43`, on `origin/main`) deleted `scoring/rating.py` (the old deterministic confidence formula) and made `InvestmentReport.confidence` an LLM-derived field.
- `src/alphaquant/agents/report_writer.py` backstory tells the LLM: *"rating and confidence must reflect the actual risk and valuation signals, not a fixed formula"* — but provides **no specific rubric**.
- Result: when given vague instructions, the LLM defaults to a mid-range number (observed: 67).
- Sub-3 real-LLM smoke (see `.superpowers/sdd/task-4-report.md`) was unable to end-to-end verify LLM confidence due to B7 (markdown-fenced JSON) + B8 (None for required fields) blockers + CrewAI 0.203.2 converter retry bug — so the 67 default cannot yet be ruled out by direct evidence.
- `src/alphaquant/scoring/dcf.py::compute_dcf_value()` exists but is unused; `scoring/rating.py` is gone. The original DCF-fix plan (`/home/lisong/.claude/plans/deep-waddling-galaxy.md`) targets an architecture that no longer exists.

User decision (2026-06-21): **fix 67 via prompt engineering + make confidence optional**, not by re-introducing deterministic rating.

## Goal

1. Give the `report_writer` LLM an explicit, criteria-based confidence rubric so numbers vary with real data and reasoning is auditable.
2. Require the LLM to document its reasoning in markdown (`## Confidence Rationale` section) so the user can see *why* a number was picked.
3. Allow `InvestmentReport.confidence` to be `None` when the LLM cannot justify a number — downstream consumers must tolerate this.

## Non-Goals (Out of Scope)

- **Real-LLM end-to-end verification** — blocked by B7/B8 + CrewAI converter bug. This spec changes code and tests; smoke verification awaits the next plan that addresses B7/B8.
- **DCF wiring into the flow** — `compute_dcf_value()` remains unused. The valuation analyst agent has its own `DCFTool`; the deterministic `scoring/dcf.py` function is dormant infrastructure. Wiring it is a separate spec.
- **Frontend / DB schema changes** — `InvestmentReport` is JSON-serialized everywhere; `confidence=None` is a natural null in JSON. Streamlit renders `confidence` via existing metrics panel; verify it handles `None` (no schema change).
- **Re-introducing deterministic rating** — explicitly rejected by user.
- **Multiple confidence fields** — single confidence is preserved.

## Design

### Change 1: `InvestmentReport.confidence` becomes optional

File: `src/alphaquant/models/report.py:33`

```python
# Before
confidence: int = Field(..., ge=0, le=100)

# After
confidence: int | None = Field(None, ge=0, le=100)
```

- Default: `None`
- Constraint retained: still `ge=0, le=100` when provided
- All other `InvestmentReport` fields unchanged

### Change 2: `report_writer` prompt gains explicit rubric

File: `src/alphaquant/agents/report_writer.py` — `backstory` string

Append a `Confidence Rubric` block to the existing backstory. The current backstory already lists required fields and the "reflect actual signals" hint; the new block makes that hint concrete.

```
Confidence Rubric (use this when picking the confidence number; if you cannot
justify a number, set confidence=null and explain in markdown):

- 80-100: Strong conviction. 5/5 data sources present (company, market, financial,
  news, competitor); DCF and relative valuation agree within 20%; risk level is
  low or medium; news sentiment is not extreme (no panic / euphoria).
- 60-79: Moderate conviction. 4/5 data sources present; DCF and relative valuation
  agree within 40%; risk is low or medium; OR one signal is weak but no major
  contradictions.
- 40-59: Low conviction. 3/5 data sources present; OR DCF vs relative diverge by
  >40%; OR risk level is high; OR news sentiment is extreme.
- 20-39: Weak conviction. 2 or fewer data sources present; OR risk level is
  extreme; OR major contradictions among the signals.
- 0-19 OR null: Cannot evaluate. Set confidence=null and document why in markdown.

Markdown must include a '## Confidence Rationale' section listing:
  - Data sources present (e.g., "5/5: company, market, financial, news, competitor")
  - DCF vs relative agreement (e.g., "DCF $180 vs relative $175, 3% spread")
  - Risk level (low / medium / high / extreme)
  - Any extreme signals
  - One-sentence verdict explaining why this confidence number (or null) was chosen.
```

Rationale: providing explicit bands + an enumeration of input signals gives the LLM a deterministic checklist rather than inviting it to guess. The markdown section makes the reasoning observable.

### Change 3: Tests updated

File: NEW `tests/test_report_optional.py` (separate from `test_models_literals.py` which is Literal-specific)

Add a `TestConfidenceOptional` class:

```python
class TestConfidenceOptional:
    def test_none_accepted(self):
        # Build a minimal InvestmentReport; only confidence-relevant field tested.
        # Use pytest.raises-style or direct construction with a stub.
        ...

    def test_zero_accepted(self):
        ...

    def test_hundred_accepted(self):
        ...

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            ...

    def test_above_100_rejected(self):
        with pytest.raises(ValidationError):
            ...
```

File: `tests/test_flow.py` — line 1014 area (existing `InvestmentReport` construction with `dcf_value=Decimal("120")`):

- If a `confidence=` field is set in that test, keep the numeric value (regression guard for non-null).
- No new test required here; the `TestConfidenceOptional` class covers the new contract.

File: `tests/test_crew.py` — search for `confidence=` usages; if any test asserts a specific confidence number derived from mock data, leave it (those numbers are from fixture, not LLM).

### Change 4: Verify downstream consumers tolerate `None`

Grep all references to `confidence` in `src/` and `src/alphaquant/frontend/`:

- `analysis_flow.py:404` — `confidence=self.state.report.confidence` — passes through unchanged; downstream uses it as-is.
- Streamlit pages — search for `.confidence` access patterns; if any use `int(report.confidence)` or arithmetic, ensure None is handled (e.g., `or 0` fallback OR show "N/A" string).

If Streamlit currently assumes `confidence` is always int, add a small guard like:
```python
confidence_display = (
    f"{report.confidence}/100" if report.confidence is not None else "N/A"
)
```

This guard is the only frontend change required, and is in-scope for this spec because otherwise the new `None` value would surface as a runtime error.

## Files Touched

| Operation | Path | Purpose |
| --- | --- | --- |
| Modify | `src/alphaquant/models/report.py` | Make `confidence` optional (`int \| None = None`) |
| Modify | `src/alphaquant/agents/report_writer.py` | Add rubric + markdown rationale block to backstory |
| Modify | `tests/test_models_literals.py` (or new `tests/test_report_optional.py`) | Add `TestConfidenceOptional` |
| Modify (potential) | `src/alphaquant/frontend/components/metrics_panel.py` or any page rendering confidence | Guard against `None` if it currently assumes int |
| New | `docs/superpowers/plans/2026-06-21-confidence-rubric.md` | Implementation plan (next skill: writing-plans) |

## Risks & Mitigations

1. **LLM still defaults to 67 with rubric** — risk: explicit bands may not change behavior if the LLM ignores structured instructions. Mitigation: rubric is structured + example-bearing; if 67 persists after this change, the next iteration can move to deterministic post-processing or per-signal decomposition.
2. **Markdown rationale free-form** — risk: LLM may emit prose that doesn't parse cleanly. Mitigation: rationale lives inside `markdown` (already required field), so format flexibility is preserved. No schema enforcement.
3. **`confidence=None` downstream** — risk: Streamlit or DB consumers crash on None. Mitigation: explicit Change 4 ensures guards exist; tests catch regressions.
4. **Test regressions** — risk: existing tests that construct `InvestmentReport` without specifying `confidence` previously failed (because field was required). Mitigation: every such test must already specify `confidence=<number>`; verify by running full test suite.

## Verification Checklist

- [ ] `InvestmentReport(confidence=None)` accepted at unit-test level
- [ ] `InvestmentReport(confidence=0)`, `confidence=50`, `confidence=100` all accepted
- [ ] `InvestmentReport(confidence=-1)` and `confidence=101` rejected with `ValidationError`
- [ ] `report_writer.py` backstory contains the rubric + markdown requirement
- [ ] All 239+ existing tests still pass (no regression)
- [ ] Streamlit (or any consumer) renders `None` without error (manual smoke or new unit test on `metrics_panel.render_metrics_panel` if feasible)
- [ ] Spec doc committed to `docs/superpowers/specs/2026-06-21-confidence-rubric-design.md`

## Open Questions for User

None. User has approved this direction (2026-06-21: "不同机制修 67" → "report_writer" → "仅 prompt 指示").