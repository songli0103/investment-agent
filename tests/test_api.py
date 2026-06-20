"""Tests for FastAPI app + CLI entry points.

The tests focus on wiring and behavior at the entry-point layer:
- App loads, has the expected routes, and the health endpoint works without
  touching the analysis Flow.
- The analyze endpoint maps domain exceptions to the correct HTTP status codes
  per spec §5.2.
- The analyze endpoint returns a well-formed AnalyzeResponse on success.
- The CLI exits with the correct status codes per the task brief.
- run_analysis / run_analysis_async wire the Flow correctly.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

# Suppress CrewAI's interactive prompt.
os.environ.setdefault("CREWAI_TESTING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from fastapi.testclient import TestClient

from alphaquant.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
)
from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    TickerNotFound,
)
from alphaquant.main import app, run_analysis, run_analysis_async
from alphaquant.models.company import Company
from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis
from alphaquant.models.report import InvestmentReport
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult


client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_report(ticker: str = "AAPL") -> InvestmentReport:
    return InvestmentReport(
        report_id="11111111-1111-1111-1111-111111111111",
        ticker=ticker,
        generated_at=datetime(2026, 6, 20),
        company=Company(
            ticker=ticker,
            name=f"{ticker} Inc.",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
        ),
        market=MarketData(
            ticker=ticker,
            as_of=datetime(2026, 6, 20),
            price=Decimal("150.00"),
            change_pct=0.5,
            volume=50_000_000,
            market_cap=3_000_000_000_000,
            pe_ratio=25.0,
            beta=1.2,
        ),
        financial=FinancialStatements(ticker=ticker),
        financial_health_score=70,
        news=NewsAnalysis(
            ticker=ticker,
            as_of=datetime(2026, 6, 20),
            total_count=10,
            positive_pct=0.5,
            negative_pct=0.2,
            neutral_pct=0.3,
            sentiment_score=0.3,
        ),
        competitors=CompetitorAnalysis(
            target_ticker=ticker,
            competitors=[
                Competitor(
                    ticker="MSFT",
                    name="Microsoft",
                    market_cap=2_500_000_000_000,
                    revenue_ttm=Decimal("200000000000"),
                )
            ],
            industry_rank=1,
            industry_size=5,
            competitive_score=75,
        ),
        risk=RiskAssessment(
            ticker=ticker,
            total_score=50,
            level="medium",
            sub_scores=[
                RiskScore(
                    category="market",
                    score=5,
                    rationale="Default neutral market risk placeholder",
                )
            ],
            top_risks=["Default neutral market risk placeholder"],
        ),
        valuation=ValuationResult(
            ticker=ticker,
            current_price=Decimal("150.00"),
            upside_pct=0.05,
            method="relative_only",
        ),
        rating="Buy",
        confidence=70,
        markdown="# Test Report",
        sources=["yahoo"],
    )


# ---------------------------------------------------------------------------
# App loading
# ---------------------------------------------------------------------------


def _app_paths(app) -> set[str]:
    """Collect every advertised path, including those from included APIRouters."""
    paths: set[str] = set()
    for r in app.router.routes:
        p = getattr(r, "path", None)
        if p:
            paths.add(p)
        # An _IncludedRouter wraps the original APIRouter; pull the prefix off
        # its include_context to find the effective (prefixed) paths.
        original = getattr(r, "original_router", None)
        include_ctx = getattr(r, "include_context", None)
        if original is not None and include_ctx is not None:
            prefix = (include_ctx.prefix or "").rstrip("/")
            for sub in original.routes:
                sp = getattr(sub, "path", None)
                if sp:
                    paths.add(f"{prefix}{sp}")
    return paths


def test_app_loads_with_expected_routes():
    """App exposes /api/v1/analyze, /api/v1/health, and FastAPI defaults."""
    paths = _app_paths(app)
    assert "/api/v1/analyze" in paths
    assert "/api/v1/health" in paths
    # FastAPI defaults
    assert "/docs" in paths
    assert "/redoc" in paths
    assert "/openapi.json" in paths


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["data_sources"] == {"yahoo": "ok", "alpha_vantage": "ok", "finnhub": "ok"}


# ---------------------------------------------------------------------------
# Analyze endpoint
# ---------------------------------------------------------------------------


def test_analyze_request_schema_rejects_bad_ticker():
    """Ticker pattern is enforced at the schema layer."""
    with pytest.raises(Exception):
        AnalyzeRequest(ticker="123")
    with pytest.raises(Exception):
        AnalyzeRequest(ticker="TOOLONG")


def test_analyze_endpoint_returns_report_on_success():
    report = _sample_report()
    with patch("alphaquant.api.routes.AnalysisFlow") as MockFlow:
        flow_instance = MockFlow.return_value
        flow_instance.state.report = report
        flow_instance.kickoff = lambda inputs: None  # no-op

        resp = client.post("/api/v1/analyze", json={"ticker": "AAPL"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["report_id"] == report.report_id
    assert body["report"]["ticker"] == "AAPL"


def test_analyze_endpoint_maps_invalid_ticker_to_400():
    # Schema-level regex requires a well-formed ticker; the Flow raises for
    # anything that *looks* valid at the schema layer but is rejected deeper.
    with patch("alphaquant.api.routes.AnalysisFlow") as MockFlow:
        flow_instance = MockFlow.return_value
        flow_instance.kickoff = lambda inputs: (_ for _ in ()).throw(
            InvalidTickerFormat("XYZ")
        )
        resp = client.post("/api/v1/analyze", json={"ticker": "XYZ"})

    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "INVALID_TICKER_FORMAT"


def test_analyze_endpoint_maps_ticker_not_found_to_404():
    with patch("alphaquant.api.routes.AnalysisFlow") as MockFlow:
        flow_instance = MockFlow.return_value
        flow_instance.kickoff = lambda inputs: (_ for _ in ()).throw(
            TickerNotFound("ZZZZ")
        )
        resp = client.post("/api/v1/analyze", json={"ticker": "ZZZZ"})

    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "TICKER_NOT_FOUND"


def test_analyze_endpoint_maps_all_sources_down_to_503():
    with patch("alphaquant.api.routes.AnalysisFlow") as MockFlow:
        flow_instance = MockFlow.return_value
        flow_instance.kickoff = lambda inputs: (_ for _ in ()).throw(
            AllDataSourcesDown("everything is down")
        )
        resp = client.post("/api/v1/analyze", json={"ticker": "AAPL"})

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "ALL_DATA_SOURCES_DOWN"


def test_analyze_endpoint_returns_500_when_no_report():
    """If the Flow runs but produces no report, we return 500 INTERNAL_ERROR."""
    with patch("alphaquant.api.routes.AnalysisFlow") as MockFlow:
        flow_instance = MockFlow.return_value
        flow_instance.state.report = None
        flow_instance.kickoff = lambda inputs: None
        resp = client.post("/api/v1/analyze", json={"ticker": "AAPL"})

    assert resp.status_code == 500
    assert resp.json()["detail"]["code"] == "INTERNAL_ERROR"


def test_analyze_response_schema():
    """AnalyzeResponse wires the nested InvestmentReport correctly."""
    report = _sample_report()
    resp = AnalyzeResponse(report_id=report.report_id, report=report)
    assert resp.status == "completed"
    assert resp.report_id == report.report_id


def test_health_response_schema():
    h = HealthResponse(
        status="ok",
        version="1.0.0",
        data_sources={"yahoo": "ok"},
    )
    assert h.status == "ok"
    assert h.data_sources == {"yahoo": "ok"}


# ---------------------------------------------------------------------------
# run_analysis / run_analysis_async
# ---------------------------------------------------------------------------


def test_run_analysis_async_returns_report():
    report = _sample_report()
    with patch("alphaquant.main.AnalysisFlow") as MockFlow:
        flow_instance = MockFlow.return_value
        flow_instance.state.report = report
        flow_instance.kickoff = lambda inputs: None

        import asyncio

        result = asyncio.run(run_analysis_async("AAPL"))

    assert result is report


def test_run_analysis_async_raises_when_no_report():
    with patch("alphaquant.main.AnalysisFlow") as MockFlow:
        flow_instance = MockFlow.return_value
        flow_instance.state.report = None
        flow_instance.kickoff = lambda inputs: None

        import asyncio

        with pytest.raises(AllDataSourcesDown):
            asyncio.run(run_analysis_async("AAPL"))


def test_run_analysis_sync_wrapper():
    """The sync entry point is a thin asyncio.run wrapper around the async one."""
    report = _sample_report()
    with patch("alphaquant.main.run_analysis_async", return_value=report) as mock:
        result = run_analysis("AAPL")
    assert result is report
    mock.assert_called_once_with("AAPL")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_usage_errors_when_no_ticker(capsys):
    from alphaquant.cli import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2  # argparse exits with 2 on usage errors


def test_cli_exit_code_for_invalid_ticker_format(capsys):
    with patch("alphaquant.cli.run_analysis", side_effect=InvalidTickerFormat("bad!")):
        code = _run_cli(["AAPL"])
    assert code == 2
    err = capsys.readouterr().err
    payload = json.loads(err.strip())
    assert payload["code"] == "INVALID_TICKER_FORMAT"


def test_cli_exit_code_for_ticker_not_found(capsys):
    with patch("alphaquant.cli.run_analysis", side_effect=TickerNotFound("ZZZZ")):
        code = _run_cli(["ZZZZ"])
    assert code == 3
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["code"] == "TICKER_NOT_FOUND"


def test_cli_exit_code_for_all_data_sources_down(capsys):
    with patch(
        "alphaquant.cli.run_analysis",
        side_effect=AllDataSourcesDown("nope"),
    ):
        code = _run_cli(["AAPL"])
    assert code == 4
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["code"] == "ALL_DATA_SOURCES_DOWN"


def test_cli_exit_code_for_unexpected_error(capsys):
    with patch("alphaquant.cli.run_analysis", side_effect=RuntimeError("boom")):
        code = _run_cli(["AAPL"])
    assert code == 1
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["code"] == "INTERNAL_ERROR"


def test_cli_writes_json_report_to_stdout(capsys):
    report = _sample_report()
    with patch("alphaquant.cli.run_analysis", return_value=report):
        code = _run_cli(["AAPL"])
    assert code == 0
    out = capsys.readouterr().out
    # JSON output should be a single line (no --pretty).
    body = json.loads(out)
    assert body["ticker"] == "AAPL"
    assert body["rating"] == "Buy"


def test_cli_pretty_flag_produces_indented_output(capsys):
    report = _sample_report()
    with patch("alphaquant.cli.run_analysis", return_value=report):
        code = _run_cli(["AAPL", "--pretty"])
    assert code == 0
    out = capsys.readouterr().out
    # Indented JSON has newlines.
    assert "\n" in out
    body = json.loads(out)
    assert body["ticker"] == "AAPL"


def test_cli_markdown_format(capsys):
    report = _sample_report()
    with patch("alphaquant.cli.run_analysis", return_value=report):
        code = _run_cli(["AAPL", "--format", "markdown"])
    assert code == 0
    out = capsys.readouterr().out
    # print() always appends a newline.
    assert out == report.markdown + "\n"


def test_cli_output_flag_writes_to_file(capsys, tmp_path):
    report = _sample_report()
    out_file = tmp_path / "report.json"
    with patch("alphaquant.cli.run_analysis", return_value=report):
        code = _run_cli(["AAPL", "--output", str(out_file)])
    assert code == 0
    body = json.loads(out_file.read_text())
    assert body["ticker"] == "AAPL"
    err = capsys.readouterr().err
    assert f"Report written to {out_file}" in err


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> int:
    """Invoke CLI main() with the given argv, bypassing argparse's sys.argv."""
    import sys

    from alphaquant.cli import main

    with patch.object(sys, "argv", ["alphaquant", *argv]):
        return main()
