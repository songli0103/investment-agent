"""测试在子项目 3 第 0 步发现的已放宽的 Pydantic 模型 Literal 约束。

子项目 3 第 0 步(AAPL 真实 LLM 运行,日志 /tmp/sub3-blocker1-aapl.log)显示
LLM 会产出非规范的 method 字符串:

- ValuationResult.method:LLM 产出 "blended" —— 原 Literal 仅允许
  ["dcf_relative_peg", "relative_only"]。已放宽以包含 "blended"、
  "dcf_only"、"relative"、"dcf_relative_blended"。

- CompetitorAnalysis.method:LLM 产出 "hybrid" —— 原 Literal 仅允许
  ["gics"、"keyword"、"manual"、"fallback"]。已放宽以包含 "hybrid"、
  "multi_factor"、"peer_comparison"。

这些测试固定放宽后的 Literal,以便在单元测试时(而非真实 LLM 运行时)
捕获未来的回归。
"""
from __future__ import annotations

import pytest
from decimal import Decimal
from pydantic import ValidationError

from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult


def _make_competitor() -> Competitor:
    """CompetitorAnalysis requires at least 1 Competitor."""
    return Competitor(
        ticker="MSFT",
        name="Microsoft",
        market_cap=2_000_000_000_000,
        revenue_ttm=Decimal("200000000000"),
    )


def _make_risk_score(category: str = "market", score: int = 5) -> RiskScore:
    """RiskScore requires rationale (min_length=10) and accepts any category string."""
    return RiskScore(
        category=category,
        score=score,
        rationale="Detailed rationale that meets the min_length=10 constraint.",
    )


def _make_risk_assessment(level: str = "medium") -> RiskAssessment:
    return RiskAssessment(
        ticker="AAPL",
        total_score=50,
        level=level,
        sub_scores=[_make_risk_score()],
        top_risks=["some risk"],
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

    def test_unknown_value_coerced_to_default(self):
        """LLM guard: unknown method strings are coerced to 'dcf_relative_peg'
        rather than crashing the flow. The Pydantic v2 Literal still constrains
        the type to str; the field_validator adds a safe-default fallback for
        unexpected LLM outputs (e.g. conversational text, 'nonsense_method_xyz')."""
        v = ValuationResult(
            ticker="AAPL",
            intrinsic_value_per_share=Decimal("150"),
            current_price=Decimal("180"),
            upside_pct=-16.67,
            method="nonsense_method_xyz",
        )
        assert v.method == "dcf_relative_peg"

    def test_verbose_llm_description_coerced_to_default(self):
        """Real failure mode observed in production: LLM returns a multi-sentence
        description instead of a Literal value. The field_validator collapses it
        to the default so the flow can continue."""
        v = ValuationResult(
            ticker="AAPL",
            intrinsic_value_per_share=Decimal("150"),
            current_price=Decimal("180"),
            upside_pct=-16.67,
            method="Multi-factor weighted DCF blended with relative comps (40%)",
        )
        assert v.method == "dcf_relative_peg"


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

    def test_unknown_value_coerced_to_default(self):
        """LLM guard: unknown method strings are coerced to 'gics' rather than
        crashing the flow. Mirrors the ValuationResult guard above; same
        pattern, different default."""
        c = CompetitorAnalysis(
            target_ticker="AAPL",
            competitors=[_make_competitor()],
            industry_rank=1,
            industry_size=5,
            competitive_score=50,
            method="nonsense_method_xyz",
        )
        assert c.method == "gics"

    def test_verbose_llm_description_coerced_to_default(self):
        """Real failure mode observed: LLM returns a verbose description
        ('Multi-factor weighted competitive adjustment (20%)') for the method
        field. Coerced to 'gics' so the flow can continue."""
        c = CompetitorAnalysis(
            target_ticker="AAPL",
            competitors=[_make_competitor()],
            industry_rank=1,
            industry_size=5,
            competitive_score=50,
            method="Multi-factor weighted competitive adjustment (20%)",
        )
        assert c.method == "gics"


class TestRiskAssessmentLevelCaseInsensitive:
    """Sub-3 Task 3 retro-fix (B4): LLM produces 'Low'/'HIGH'; normalize to lowercase."""

    @pytest.mark.parametrize("input_level", ["low", "medium", "high", "extreme"])
    def test_lowercase_accepted(self, input_level):
        ra = _make_risk_assessment(level=input_level)
        assert ra.level == input_level

    @pytest.mark.parametrize(
        "input_level,expected",
        [
            ("Low", "low"),
            ("LOW", "low"),
            ("Medium", "medium"),
            ("HIGH", "high"),
            ("Extreme", "extreme"),
            ("LoW", "low"),
        ],
    )
    def test_capitalized_normalized_to_lowercase(self, input_level, expected):
        ra = _make_risk_assessment(level=input_level)
        assert ra.level == expected

    def test_truly_unknown_value_rejected(self):
        """Sanity: a non-Literal value still rejected (just lowercased first)."""
        with pytest.raises(ValidationError):
            _make_risk_assessment(level="catastrophic")


class TestRiskScoreCategoryAnyString:
    """Sub-3 Task 3 retro-fix (B5): LLM produces human-readable strings."""

    @pytest.mark.parametrize(
        "category",
        [
            "Market Risk",
            "Credit Risk",
            "Operational Risk",
            "Liquidity Risk",
            "Regulatory/Compliance Risk",
            "Volatility Risk",
            "financial",  # old canonical still works
            "market",
            "anything_goes_xyz",
        ],
    )
    def test_any_string_accepted(self, category):
        rs = _make_risk_score(category=category)
        assert rs.category == category


class TestRiskScoreScoreRange:
    """Sub-3 Task 3 retro-fix (B6): widened 0-10 -> 0-100."""

    @pytest.mark.parametrize("score", [0, 5, 10, 15, 30, 50, 100])
    def test_widened_range_accepted(self, score):
        rs = _make_risk_score(score=score)
        assert rs.score == score

    def test_negative_still_rejected(self):
        with pytest.raises(ValidationError):
            _make_risk_score(score=-1)

    def test_above_100_still_rejected(self):
        with pytest.raises(ValidationError):
            _make_risk_score(score=101)