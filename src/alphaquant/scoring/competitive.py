"""Competitive score (0–100) based on percentile vs peers."""
from __future__ import annotations

from statistics import mean

from alphaquant.models.competitor import Competitor, CompetitorAnalysis


def _percentile_rank(value: float, peer_values: list[float]) -> float:
    """Percentile rank 0-100."""
    if not peer_values:
        return 50.0
    below = sum(1 for v in peer_values if v < value)
    return (below / len(peer_values)) * 100


def compute(target_metrics: dict[str, float], peers: list[Competitor]) -> int:
    """Score target by averaging percentile rank across dimensions."""
    dimensions = ["market_cap", "revenue_growth_yoy", "gross_margin", "net_margin"]
    percentile_scores: list[float] = []
    for dim in dimensions:
        target_val = target_metrics.get(dim)
        peer_vals = [
            getattr(p, dim) for p in peers if getattr(p, dim, None) is not None
        ]
        if target_val is None or not peer_vals:
            continue
        percentile_scores.append(_percentile_rank(float(target_val), [float(v) for v in peer_vals]))
    if not percentile_scores:
        return 50
    return round(mean(percentile_scores))
