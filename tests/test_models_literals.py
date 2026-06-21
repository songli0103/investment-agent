"""Test widened Pydantic model Literal constraints discovered during sub-3 Step 0.

Sub-3 Step 0 (AAPL real LLM run, log /tmp/sub3-blocker1-aapl.log) showed the LLM
producing non-canonical method strings:

- ValuationResult.method: LLM produced "blended" — original Literal only allowed
  ["dcf_relative_peg", "relative_only"]. Widened to include "blended",
  "dcf_only", "relative", "dcf_relative_blended".

- CompetitorAnalysis.method: LLM produced "hybrid" — original Literal only allowed
  ["gics", "keyword", "manual", "fallback"]. Widened to include "hybrid",
  "multi_factor", "peer_comparison".

These tests pin the widened Literals so future regressions are caught at unit-test
time rather than at real-LLM runtime.
"""
from __future__ import annotations

import pytest
from decimal import Decimal
from pydantic import ValidationError

from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.valuation import ValuationResult


def _make_competitor() -> Competitor:
    """CompetitorAnalysis requires at least 1 Competitor."""
    return Competitor(
        ticker="MSFT",
        name="Microsoft",
        market_cap=2_000_000_000_000,
        revenue_ttm=Decimal("200000000000"),
    )


class TestValuationResultMethodLiteral:
    def test_original_values_still_accepted(self):
        """Backward compatibility: the two pre-existing Literal values still work."""
        for method in ("dcf_relative_peg", "relative_only"):
            v = ValuationResult(
                ticker="AAPL",
                intrinsic_value_per_share=Decimal("150"),
                current_price=Decimal("180"),
                upside_pct=-16.67,
                method=method,
            )
            assert v.method == method

    def test_blended_accepted(self):
        """Sub-3 Step 0: LLM produced 'blended' for AAPL. Must be accepted."""
        v = ValuationResult(
            ticker="AAPL",
            intrinsic_value_per_share=Decimal("150"),
            current_price=Decimal("180"),
            upside_pct=-16.67,
            method="blended",
        )
        assert v.method == "blended"

    @pytest.mark.parametrize("method", ["dcf_only", "relative", "dcf_relative_blended"])
    def test_widened_values_accepted(self, method):
        """Other realistic method strings the LLM may produce."""
        v = ValuationResult(
            ticker="AAPL",
            intrinsic_value_per_share=Decimal("150"),
            current_price=Decimal("180"),
            upside_pct=-16.67,
            method=method,
        )
        assert v.method == method

    def test_unknown_value_rejected(self):
        """Sanity: truly unknown values are still rejected (Literal still constrained)."""
        with pytest.raises(ValidationError):
            ValuationResult(
                ticker="AAPL",
                intrinsic_value_per_share=Decimal("150"),
                current_price=Decimal("180"),
                upside_pct=-16.67,
                method="nonsense_method_xyz",
            )


class TestCompetitorAnalysisMethodLiteral:
    def test_original_values_still_accepted(self):
        """Backward compatibility."""
        for method in ("gics", "keyword", "manual", "fallback"):
            c = CompetitorAnalysis(
                target_ticker="AAPL",
                competitors=[_make_competitor()],
                industry_rank=1,
                industry_size=5,
                competitive_score=50,
                method=method,
            )
            assert c.method == method

    def test_hybrid_accepted(self):
        """Sub-3 Step 0: LLM produced 'hybrid' for AAPL. Must be accepted."""
        c = CompetitorAnalysis(
            target_ticker="AAPL",
            competitors=[_make_competitor()],
            industry_rank=1,
            industry_size=5,
            competitive_score=50,
            method="hybrid",
        )
        assert c.method == "hybrid"

    @pytest.mark.parametrize("method", ["multi_factor", "peer_comparison"])
    def test_widened_values_accepted(self, method):
        """Other realistic method strings."""
        c = CompetitorAnalysis(
            target_ticker="AAPL",
            competitors=[_make_competitor()],
            industry_rank=1,
            industry_size=5,
            competitive_score=50,
            method=method,
        )
        assert c.method == method

    def test_unknown_value_rejected(self):
        with pytest.raises(ValidationError):
            CompetitorAnalysis(
                target_ticker="AAPL",
                competitors=[_make_competitor()],
                industry_rank=1,
                industry_size=5,
                competitive_score=50,
                method="nonsense_method_xyz",
            )