"""CLI entry: `python -m alphaquant AAPL`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    TickerNotFound,
)
from alphaquant.core import run_analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="alphaquant",
        description="AI Investment Research Analyst",
    )
    parser.add_argument("ticker", help="US stock ticker (e.g. AAPL)")
    parser.add_argument(
        "--format", choices=["json", "markdown"], default="json", help="Output format"
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--output", type=str, help="Write to file instead of stdout")
    args = parser.parse_args()

    try:
        report = run_analysis(args.ticker)
    except InvalidTickerFormat as e:
        print(json.dumps({"code": "INVALID_TICKER_FORMAT", "message": str(e)}), file=sys.stderr)
        return 2
    except TickerNotFound as e:
        print(json.dumps({"code": "TICKER_NOT_FOUND", "message": str(e)}), file=sys.stderr)
        return 3
    except AllDataSourcesDown as e:
        print(json.dumps({"code": "ALL_DATA_SOURCES_DOWN", "message": str(e)}), file=sys.stderr)
        return 4
    except Exception as e:
        print(json.dumps({"code": "INTERNAL_ERROR", "message": str(e)}), file=sys.stderr)
        return 1

    if args.format == "json":
        output = report.model_dump_json(indent=2 if args.pretty else None)
    else:
        output = report.markdown

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
