"""ReportWriter Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM


def build_report_writer_agent(llm: LLM) -> Agent:
    return Agent(
        role="Investment Report Synthesizer",
        goal=(
            "Synthesize upstream analyses (competitor summary, risk summary, "
            "valuation summary) and data (company, market, financial, news) into "
            "a final investment report markdown plus a rating, confidence, "
            "horizon, and catalysts list."
        ),
        backstory=(
            "You are a senior investment writer. The Flow already computes the "
            "structured competitor/risk/valuation analyses from data and the "
            "upstream 3 analysis agents contribute text summaries as context. "
            "You output a slim Pydantic ReportWriterOutput object with these "
            "fields: rating (one of 'Strong Buy'|'Buy'|'Hold'|'Sell'|'Strong Sell'), "
            "confidence (0-100, nullable), investment_horizon "
            "('short'|'medium'|'long'), catalysts (≥1 short bullet), and markdown "
            "(≥100 chars, structured sections). "
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
            "chosen."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_report_writer_agent"]