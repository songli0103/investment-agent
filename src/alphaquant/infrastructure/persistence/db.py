"""AlphaQuant 报告历史的 SQLite 持久化层。"""
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
    confidence INTEGER,
    market_price REAL,
    report_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_ticker ON reports(ticker);
CREATE INDEX IF NOT EXISTS idx_reports_generated_at ON reports(generated_at);
"""

# 为 confidence 变为可空之前创建的 DB 进行迁移。SQLite >=3.35 支持
# ALTER TABLE ... ALTER COLUMN ... DROP NOT NULL;对较老版本则回退到重建表。
_MIGRATION_V2 = "ALTER TABLE reports ALTER COLUMN confidence DROP NOT NULL;"


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """幂等地运行加性迁移。"""
    cur = conn.execute("PRAGMA user_version")
    version = int(cur.fetchone()[0])
    if version < 2:
        try:
            conn.execute(_MIGRATION_V2)
            conn.execute("PRAGMA user_version = 2")
        except sqlite3.OperationalError:
            # 较老版本 SQLite:回退到重建表。
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS reports_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    confidence INTEGER,
                    market_price REAL,
                    report_json TEXT NOT NULL
                );
                INSERT INTO reports_new
                    SELECT id, ticker, generated_at, rating, confidence,
                           market_price, report_json FROM reports;
                DROP TABLE reports;
                ALTER TABLE reports_new RENAME TO reports;
                CREATE INDEX IF NOT EXISTS idx_reports_ticker
                    ON reports(ticker);
                CREATE INDEX IF NOT EXISTS idx_reports_generated_at
                    ON reports(generated_at);
                PRAGMA user_version = 2;
                """
            )
        conn.commit()


class DB:
    """用于存储 InvestmentReport 历史的轻量级 sqlite3 包装。"""

    def __init__(self, path: str | Path = "./data/reports.db") -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        """如果 reports 表和索引不存在则创建,然后运行迁移。"""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            _apply_migrations(conn)
            conn.commit()

    def insert_report(self, ticker: str, report: InvestmentReport) -> int:
        """持久化完整报告,返回新行的 id。"""
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
        """返回去重后的 ticker,按升序排序。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM reports ORDER BY ticker ASC"
            ).fetchall()
        return [row["ticker"] for row in rows]

    def get_latest_report(self, ticker: str | None = None) -> ReportRecord | None:
        """返回全局最近一条 ``ReportRecord``,或指定 ``ticker`` 的最近一条。

        用于 Analyze 页面在浏览器刷新后自动恢复上次的分析结果。
        数据库为空时返回 ``None``。
        """
        sql = (
            "SELECT id, ticker, generated_at, rating, confidence, "
            "market_price, report_json FROM reports "
        )
        params: list[object] = []
        if ticker is not None:
            sql += "WHERE ticker = ? "
            params.append(ticker)
        sql += "ORDER BY generated_at DESC, id DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return self._row_to_record(row) if row else None

    def get_latest_reports(self, limit: int = 5) -> list[ReportRecord]:
        """返回全局最近的 ``limit`` 条报告(不带 ticker 过滤)。

        不同的 ``(ticker, generated_at)`` 对,最新优先,用于 Analyze 页面
        在页面加载时显示"最近分析"面板。
        """
        if limit <= 0:
            return []
        sql = (
            "SELECT id, ticker, generated_at, rating, confidence, "
            "market_price, report_json FROM reports "
            "ORDER BY generated_at DESC, id DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, [limit]).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_history(
        self,
        tickers: list[str] | None = None,
        since: datetime | None = None,
    ) -> list[ReportRecord]:
        """返回按 ticker 列表和/或最小 generated_at 过滤的报告行。"""
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
        """删除所有行,返回删除的数量。"""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM reports")
            conn.commit()
            return cur.rowcount

    def count(self) -> int:
        """返回存储报告的总数。"""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM reports").fetchone()
        return int(row["n"])

    def export_jsonl(self) -> Iterator[str]:
        """为每条存储的报告产生一行 JSON。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT report_json FROM reports ORDER BY id ASC"
            ).fetchall()
        for row in rows:
            # 校验存储的 JSON 可解析(尽早发现损坏)。
            payload = json.loads(row["report_json"])
            yield json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ReportRecord:
        return ReportRecord(
            id=int(row["id"]),
            ticker=row["ticker"],
            generated_at=datetime.fromisoformat(row["generated_at"]),
            rating=row["rating"],
            confidence=(
                int(row["confidence"]) if row["confidence"] is not None else None
            ),
            market_price=(
                float(row["market_price"]) if row["market_price"] is not None else None
            ),
            report_json=row["report_json"],
        )