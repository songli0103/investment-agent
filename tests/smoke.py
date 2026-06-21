"""Smoke test for AlphaQuant MVP.

Usage: python -m tests.smoke

Asserts:
  1. Flow runs without crash
  2. Output is InvestmentReport instance
  3. All 8 sections populated
  4. Rating is in valid set
  5. Confidence 0-100
  6. Markdown > 500 chars
  7. Sources list non-empty
"""
from __future__ import annotations

import asyncio
import sys

from alphaquant.models.report import InvestmentReport
from alphaquant.main import run_analysis_async
from alphaquant.observability import configure_logging

SMOKE_TICKER = "AAPL"
REQUIRED_SECTIONS = [
    "company",
    "market",
    "financial",
    "news",
    "competitors",
    "risk",
    "valuation",
    "rating",
]

configure_logging()


def assert_(condition: bool, msg: str) -> None:
    if not condition:
        print(f"❌ FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"✅ {msg}")


def main() -> int:
    print(f"=== AlphaQuant Smoke Test ===")
    print(f"Ticker: {SMOKE_TICKER}\n")

    # Assertion 1: Flow runs without crash
    print("→ Running AnalysisFlow...")
    try:
        report = asyncio.run(run_analysis_async(SMOKE_TICKER))
    except Exception as e:
        print(f"❌ Flow crashed: {e}", file=sys.stderr)
        return 1
    assert_(True, "Flow ran without crash")

    # Assertion 2: Pydantic schema valid
    assert_(isinstance(report, InvestmentReport), "Output is InvestmentReport")

    # Assertion 3: All sections populated
    print("\n→ Checking required sections...")
    rd = report.model_dump()
    for section in REQUIRED_SECTIONS:
        assert_(section in rd and rd[section] is not None, f"Section '{section}' present")

    # Assertion 4: Rating valid
    valid_ratings = {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}
    assert_(report.rating in valid_ratings, f"Rating '{report.rating}' is valid")

    # Assertion 5: Confidence range
    if report.confidence is not None:
        assert_(0 <= report.confidence <= 100, f"Confidence {report.confidence} in [0,100]")

    # Assertion 6: Markdown non-trivial
    assert_(len(report.markdown) > 500, f"Markdown has {len(report.markdown)} chars (>500)")

    # Assertion 7: Sources cited (and not just the "degraded" status marker).
    sources = [s for s in report.sources if s != "degraded"]
    assert_(
        len(sources) >= 1,
        f"Report cites {len(sources)} non-degraded sources: {report.sources}",
    )

    print(f"\n=== ✅ SMOKE PASSED ===")
    print(f"Report ID: {report.report_id}")
    print(f"Rating: {report.rating} (confidence {report.confidence}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
