"""CrewAI Agent definitions."""

from alphaquant.agents.company_resolver import build_company_resolver_agent
from alphaquant.agents.competitor_analyst import build_competitor_analyst_agent
from alphaquant.agents.financial_analyst import build_financial_analyst_agent
from alphaquant.agents.market_analyst import build_market_analyst_agent
from alphaquant.agents.news_analyst import build_news_analyst_agent
from alphaquant.agents.report_writer import build_report_writer_agent
from alphaquant.agents.risk_analyst import build_risk_analyst_agent
from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent

__all__ = [
    "build_company_resolver_agent",
    "build_competitor_analyst_agent",
    "build_financial_analyst_agent",
    "build_market_analyst_agent",
    "build_news_analyst_agent",
    "build_report_writer_agent",
    "build_risk_analyst_agent",
    "build_valuation_analyst_agent",
]