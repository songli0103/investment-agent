"""Risk score (0–100, higher = riskier)."""
from __future__ import annotations

from alphaquant.models.risk import RiskScore


WEIGHTS = {
    "financial": 0.30,
    "operational": 0.15,
    "market": 0.15,
    "regulatory": 0.15,
    "governance": 0.10,
    "macro": 0.15,
}


def compute(sub_scores: list[RiskScore]) -> int:
    """Total risk score 0-100 from sub-scores 0-10."""
    if not sub_scores:
        return 50
    weighted_sum = sum(s.score * 10 * WEIGHTS.get(s.category, 0.1) for s in sub_scores)
    return min(100, max(0, round(weighted_sum)))


def determine_level(total_score: int) -> str:
    """Map 0-100 to risk level."""
    if total_score <= 25:
        return "low"
    if total_score <= 50:
        return "medium"
    if total_score <= 75:
        return "high"
    return "extreme"
