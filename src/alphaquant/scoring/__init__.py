"""Scoring module — only deterministic helpers used by tools and LLM tools remain.

Sub-project 3: `rating`, `competitive`, `risk_score` modules removed (LLM-driven now).
`dcf` and `financial_health` remain because the ValuationAnalyst and ReportWriter
agents can call them as tools during reasoning.
"""
from alphaquant.scoring import dcf, financial_health

__all__ = ["dcf", "financial_health"]