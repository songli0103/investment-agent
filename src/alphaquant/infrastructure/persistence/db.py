"""SQLite persistence layer for AlphaQuant report history."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from alphaquant.infrastructure.persistence.models import ReportRecord
from alphaquant.models.report import InvestmentReport


SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    rating TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    market_price REAL,
    report_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_ticker ON reports(ticker);
CREATE INDEX IF NOT EXISTS idx_reports_generated_at ON reports(generated_at);
"""


class DB:
    """Thin sqlite3 wrapper for storing InvestmentReport history."""

    def __init__(self, path: str | Path = "./data/reports.db") -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        """Create reports table and indexes if absent."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def insert_report(self, ticker: str, report: InvestmentReport) -> int:
        """Persist a full report, returning the new row id."""
        report_json = report.model_dump_json()
        market_price = float(report.market.price) if report.market is not None else None
        generated_at = report.generated_at.isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO reports (
                    ticker, generated_at, rating, confidence, market_price, report_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    generated_at,
                    report.rating,
                    report.confidence,
                    market_price,
                    report_json,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_tickers(self) -> list[str]:
        """Return distinct tickers, sorted ascending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM reports ORDER BY ticker ASC"
            ).fetchall()
        return [row["ticker"] for row in rows]

    def get_history(
        self,
        tickers: list[str] | None = None,
        since: datetime | None = None,
    ) -> list[ReportRecord]:
        """Return report rows filtered by ticker list and/or minimum generated_at."""
        clauses: list[str] = []
        params: list[object] = []
        if tickers:
            placeholders = ",".join("?" for _ in tickers)
            clauses.append(f"ticker IN ({placeholders})")
            params.extend(tickers)
        if since is not None:
            clauses.append("generated_at >= ?")
            params.append(since.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT id, ticker, generated_at, rating, confidence, "
            "market_price, report_json "
            f"FROM reports {where} ORDER BY generated_at ASC, id ASC"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete_all(self) -> int:
        """Delete every row, returning the number deleted."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM reports")
            conn.commit()
            return cur.rowcount

    def count(self) -> int:
        """Return the total number of stored reports."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM reports").fetchone()
        return int(row["n"])

    def export_jsonl(self) -> Iterator[str]:
        """Yield one JSON line per stored report."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT report_json FROM reports ORDER BY id ASC"
            ).fetchall()
        for row in rows:
            # Validate that stored JSON parses (catches corruption early).
            payload = json.loads(row["report_json"])
            yield json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ReportRecord:
        return ReportRecord(
            id=int(row["id"]),
            ticker=row["ticker"],
            generated_at=datetime.fromisoformat(row["generated_at"]),
            rating=row["rating"],
            confidence=int(row["confidence"]),
            market_price=(
                float(row["market_price"]) if row["market_price"] is not None else None
            ),
            report_json=row["report_json"],
        )