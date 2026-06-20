"""Investment rating logic (5-level)."""
from __future__ import annotations

from alphaquant.models.news import NewsAnalysis
from alphaquant.models.risk import RiskAssessment
from alphaquant.models.valuation import ValuationResult


def determine_rating(
    valuation: ValuationResult,
    risk: RiskAssessment,
    sentiment: NewsAnalysis,
) -> tuple[str, int]:
    """Returns (rating, confidence)."""
    upside = valuation.upside_pct

    # Base rating by upside
    if upside >= 0.30:
        base = "Strong Buy"
    elif upside >= 0.10:
        base = "Buy"
    elif upside > -0.10:
        base = "Hold"
    elif upside > -0.30:
        base = "Sell"
    else:
        base = "Strong Sell"

    # Risk adjustment: downgrade bullish ratings if risk high
    if risk.level in ("high", "extreme") and base in ("Strong Buy", "Buy"):
        base = "Hold"

    # Sentiment adjustment: downgrade if strongly negative
    if sentiment.sentiment_score < -0.5 and base in ("Strong Buy", "Buy"):
        base = "Hold"

    confidence = _compute_confidence(valuation, risk, sentiment)
    return base, confidence


def _compute_confidence(
    valuation: ValuationResult,
    risk: RiskAssessment,
    sentiment: NewsAnalysis,
) -> int:
    """0-100 confidence based on data completeness and signal alignment."""
    data_completeness = 50
    if valuation.dcf_value is not None:
        data_completeness += 20
    if valuation.relative_value is not None:
        data_completeness += 15
    if sentiment.total_count > 5:
        data_completeness += 15

    method_coverage = 30
    if valuation.dcf_value is not None:
        method_coverage += 35
    if valuation.relative_value is not None:
        method_coverage += 35

    # Signal alignment (simplified)
    signal_alignment = 70  # base
    if risk.level in ("high", "extreme"):
        signal_alignment -= 20
    if sentiment.sentiment_score < -0.3:
        signal_alignment -= 10
    if sentiment.sentiment_score > 0.3:
        signal_alignment += 10

    return min(100, max(0, round(data_completeness * 0.4 + method_coverage * 0.2 + signal_alignment * 0.4)))
