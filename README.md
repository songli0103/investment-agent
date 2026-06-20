# AlphaQuant

AI-powered investment research analyst. Takes a US stock ticker, returns a structured investment report in ~30 seconds.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) — fast Python package manager
- `MINIMAX_API_KEY` (required)
- Optional: `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `NEWS_API_KEY`

## Install

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone, sync deps, create .venv
git clone git@github.com:songli0103/investment-agent.git
cd investment-agent
uv sync --extra dev

# Configure environment
cp .env.example .env
# Edit .env to set MINIMAX_API_KEY
```

`uv sync` automatically creates a `.venv` and installs everything pinned in `uv.lock`.

## Run

All commands use `uv run` (which activates `.venv` on the fly).

### CLI

```bash
uv run python -m alphaquant AAPL
uv run python -m alphaquant AAPL --format markdown
uv run python -m alphaquant AAPL --pretty --output report.json
```

### FastAPI

```bash
uv run uvicorn alphaquant.main:app --reload
```

Then:
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

Docs at http://localhost:8000/docs

> The in-process rate limiter is per-worker — run with `--workers 1` for now, or replace with a shared limiter before scaling out.

## Test

```bash
uv run pytest -q              # Full suite (179 tests)
uv run python -m tests.smoke  # End-to-end smoke test (7 assertions)
```

## Architecture

8 CrewAI Agents orchestrated via Flow:
1. CompanyResolver → 2. MarketAnalyst (parallel with 3, 4) → 3. NewsAnalyst → 4. FinancialAnalyst → 5. CompetitorAnalyst → 6. RiskAnalyst → 7. ValuationAnalyst → 8. ReportWriter

Data sources (fallback chain): Yahoo Finance → Alpha Vantage → Finnhub → SEC EDGAR → NewsAPI.

LLM: MiniMax-M3 via LiteLLM.