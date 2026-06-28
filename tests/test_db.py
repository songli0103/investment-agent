"""SQLite 持久化层的测试。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from alphaquant.infrastructure.persistence import DB, ReportRecord
from alphaquant.models.company import Company
from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis
from alphaquant.models.report import InvestmentReport
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(
    ticker: str = "AAPL",
    generated_at: datetime | None = None,
    rating: str = "Buy",
    confidence: int = 70,
    price: str = "150.00",
    report_id: str = "11111111-1111-1111-1111-111111111111",
) -> InvestmentReport:
    return InvestmentReport(
        report_id=report_id,
        ticker=ticker,
        generated_at=generated_at or datetime(2026, 6, 20),
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
            as_of=generated_at or datetime(2026, 6, 20),
            price=Decimal(price),
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
            as_of=generated_at or datetime(2026, 6, 20),
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
            current_price=Decimal(price),
            upside_pct=10.0,
        ),
        rating=rating,
        confidence=confidence,
        markdown=f"# {ticker}\n\nSample report.",
    )


@pytest.fixture()
def db(tmp_path) -> DB:
    instance = DB(tmp_path / "reports.db")
    instance.init()
    return instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_creates_table(tmp_path) -> None:
    db_path = tmp_path / "fresh.db"
    db = DB(db_path)
    assert not db_path.exists() or db_path.stat().st_size == 0

    db.init()

    assert db_path.exists()
    with db._connect() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reports'"
        ).fetchone()
    assert row is not None
    assert row["name"] == "reports"


def test_insert_and_get_history(db: DB) -> None:
    ids = [
        db.insert_report(
            "AAPL",
            _make_report(
                ticker="AAPL",
                generated_at=datetime(2026, 6, 18),
                report_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            ),
        ),
        db.insert_report(
            "AAPL",
            _make_report(
                ticker="AAPL",
                generated_at=datetime(2026, 6, 19),
                report_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            ),
        ),
        db.insert_report(
            "AAPL",
            _make_report(
                ticker="AAPL",
                generated_at=datetime(2026, 6, 20),
                report_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
            ),
        ),
    ]
    assert ids == sorted(ids)
    assert ids[0] >= 1

    history = db.get_history()
    assert len(history) == 3
    assert [r.ticker for r in history] == ["AAPL", "AAPL", "AAPL"]
    assert [r.generated_at for r in history] == sorted(r.generated_at for r in history)
    for record in history:
        assert isinstance(record, ReportRecord)
        assert record.rating == "Buy"
        assert record.confidence == 70
        assert record.market_price == pytest.approx(150.0)
        # Full JSON payload is round-trippable.
        parsed = InvestmentReport.model_validate_json(record.report_json)
        assert parsed.ticker == "AAPL"


def test_list_tickers(db: DB) -> None:
    # 3 MSFT + 2 AAPL, deliberately out of alphabetical order.
    for i, ts in enumerate(
        [
            datetime(2026, 6, 18),
            datetime(2026, 6, 19),
            datetime(2026, 6, 20),
        ]
    ):
        db.insert_report("MSFT", _make_report(ticker="MSFT", generated_at=ts, price="300.00"))
    for ts in [datetime(2026, 6, 18), datetime(2026, 6, 19)]:
        db.insert_report("AAPL", _make_report(ticker="AAPL", generated_at=ts))

    assert db.list_tickers() == ["AAPL", "MSFT"]


def test_get_history_filters_by_ticker_and_since(db: DB) -> None:
    for day in range(1, 6):
        ts = datetime(2026, 6, day)
        db.insert_report(
            "AAPL",
            _make_report(
                ticker="AAPL",
                generated_at=ts,
                report_id=f"00000000-0000-0000-0000-{day:012d}",
            ),
        )
    for day in range(1, 6):
        ts = datetime(2026, 6, day)
        db.insert_report(
            "MSFT",
            _make_report(
                ticker="MSFT",
                generated_at=ts,
                price="300.00",
                report_id=f"11111111-0000-0000-0000-{day:012d}",
            ),
        )

    only_aapl = db.get_history(tickers=["AAPL"])
    assert {r.ticker for r in only_aapl} == {"AAPL"}
    assert len(only_aapl) == 5

    since = datetime(2026, 6, 3)
    since_records = db.get_history(tickers=["MSFT"], since=since)
    assert [r.generated_at for r in since_records] == [
        datetime(2026, 6, 3),
        datetime(2026, 6, 4),
        datetime(2026, 6, 5),
    ]

    # Since applies on its own (all tickers).
    all_since = db.get_history(since=datetime(2026, 6, 5))
    assert len(all_since) == 2  # 1 AAPL + 1 MSFT on day 5


def test_delete_all(db: DB) -> None:
    for day in range(1, 4):
        db.insert_report(
            "AAPL",
            _make_report(ticker="AAPL", generated_at=datetime(2026, 6, day)),
        )
    assert db.count() == 3

    deleted = db.delete_all()

    assert deleted == 3
    assert db.count() == 0
    assert db.list_tickers() == []
    assert db.get_history() == []


def test_export_jsonl(db: DB) -> None:
    for i, day in enumerate([1, 2, 3], start=1):
        db.insert_report(
            "AAPL",
            _make_report(
                ticker="AAPL",
                generated_at=datetime(2026, 6, day),
                report_id=f"00000000-0000-0000-0000-{i:012d}",
            ),
        )

    lines = list(db.export_jsonl())

    assert len(lines) == 3
    for line in lines:
        payload = json.loads(line)
        assert payload["ticker"] == "AAPL"
        assert payload["rating"] == "Buy"
        assert payload["confidence"] == 70
        # Each line must be valid JSON; newlines inside the body shouldn't split rows.
        assert "\n" not in line


def test_concurrent_inserts(db: DB) -> None:
    inserted_ids: list[int] = []
    for i in range(10):
        rid = db.insert_report(
            "AAPL",
            _make_report(
                ticker="AAPL",
                generated_at=datetime(2026, 6, 1) + timedelta(days=i),
                report_id=f"00000000-0000-0000-0000-{i:012d}",
            ),
        )
        inserted_ids.append(rid)

    assert len(set(inserted_ids)) == 10  # all unique ids
    assert db.count() == 10

    history = db.get_history()
    assert len(history) == 10
    # All ids retrieved match inserted ids.
    assert sorted(r.id for r in history) == sorted(inserted_ids)
    # Order is chronological (by generated_at then id).
    generated = [r.generated_at for r in history]
    assert generated == sorted(generated)


class TestNullableConfidence:
    def test_insert_and_read_with_null_confidence(self, tmp_path, stub_report):
        """Sub-plan: confidence can be null in InvestmentReport -> DB -> row."""
        db_path = tmp_path / "null_conf.db"
        db = DB(db_path)
        db.init()
        rep = stub_report(
            report_id="00000000-0000-0000-0000-000000000001",
            ticker="TEST",
            confidence=None,  # <-- the change under test
        )
        new_id = db.insert_report("TEST", rep)
        rows = db.get_history()
        assert len(rows) == 1
        assert rows[0].id == new_id
        assert rows[0].confidence is None

    def test_insert_and_read_with_numeric_confidence(self, tmp_path, stub_report):
        """Regression: numeric confidence still round-trips."""
        db_path = tmp_path / "num_conf.db"
        db = DB(db_path)
        db.init()
        rep = stub_report(
            report_id="00000000-0000-0000-0000-000000000002",
            ticker="TEST",
            confidence=80,
        )
        db.insert_report("TEST", rep)
        rows = db.get_history()
        assert rows[0].confidence == 80


# ---------------------------------------------------------------------------
# get_latest_report / get_latest_reports
# ---------------------------------------------------------------------------


def test_get_latest_report_empty_db(tmp_path) -> None:
    """get_latest_report on an empty DB must return None, not raise."""
    db = DB(tmp_path / "empty.db")
    db.init()
    assert db.get_latest_report() is None
    assert db.get_latest_report(ticker="AAPL") is None


def test_get_latest_report_returns_most_recent(tmp_path) -> None:
    """get_latest_report orders by generated_at DESC, id DESC."""
    db = DB(tmp_path / "latest.db")
    db.init()
    older = _make_report(
        report_id="11111111-1111-1111-1111-111111111111",
        ticker="AAPL",
        generated_at=datetime(2026, 6, 20, 10, 0, 0),
    )
    newer = _make_report(
        report_id="22222222-2222-2222-2222-222222222222",
        ticker="AAPL",
        generated_at=datetime(2026, 6, 21, 10, 0, 0),
        rating="Hold",
    )
    db.insert_report("AAPL", older)
    db.insert_report("AAPL", newer)
    latest = db.get_latest_report()
    assert latest is not None
    assert latest.rating == "Hold"
    # ReportRecord stores report_id inside the JSON; deserialize to confirm.
    parsed = InvestmentReport.model_validate_json(latest.report_json)
    assert parsed.report_id == "22222222-2222-2222-2222-222222222222"


def test_get_latest_report_filters_by_ticker(tmp_path) -> None:
    """Ticker filter ignores other symbols even if newer."""
    db = DB(tmp_path / "filter.db")
    db.init()
    db.insert_report(
        "MSFT",
        _make_report(
            report_id="33333333-3333-3333-3333-333333333333",
            ticker="MSFT",
            generated_at=datetime(2026, 6, 22, 10, 0, 0),
        ),
    )
    db.insert_report(
        "AAPL",
        _make_report(
            report_id="44444444-4444-4444-4444-444444444444",
            ticker="AAPL",
            generated_at=datetime(2026, 6, 20, 10, 0, 0),
        ),
    )
    aapl = db.get_latest_report(ticker="AAPL")
    assert aapl is not None
    assert aapl.ticker == "AAPL"
    parsed = InvestmentReport.model_validate_json(aapl.report_json)
    assert parsed.report_id == "44444444-4444-4444-4444-444444444444"


def test_get_latest_reports_caps_at_limit(tmp_path) -> None:
    """get_latest_reports(limit=N) returns at most N rows, newest first."""
    db = DB(tmp_path / "recent.db")
    db.init()
    expected_ids: list[str] = []
    for i in range(7):
        rid = f"55555555-5555-5555-5555-{i:012d}"
        expected_ids.append(rid)
        db.insert_report(
            "AAPL",
            _make_report(
                report_id=rid,
                ticker="AAPL",
                generated_at=datetime(2026, 6, 15) + timedelta(hours=i),
            ),
        )
    rows = db.get_latest_reports(limit=3)
    assert len(rows) == 3
    # Newest first: i=6, i=5, i=4
    parsed_ids = [
        InvestmentReport.model_validate_json(r.report_json).report_id
        for r in rows
    ]
    assert parsed_ids == [expected_ids[6], expected_ids[5], expected_ids[4]]


def test_get_latest_reports_zero_limit_returns_empty(tmp_path) -> None:
    """Edge case: limit=0 returns no rows (not all rows)."""
    db = DB(tmp_path / "zero.db")
    db.init()
    db.insert_report("AAPL", _make_report())
    assert db.get_latest_reports(limit=0) == []