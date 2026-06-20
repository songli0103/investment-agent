# AlphaQuant

AI-powered investment research analyst. Takes a US stock ticker, returns a structured investment report in ~30 seconds.

## Requirements

- Python 3.11+
- `MINIMAX_API_KEY` (required)
- Optional: `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `NEWS_API_KEY`

## Install

```bash
pip install -e ".[dev]"
cp .env.example .env
# Edit .env to set MINIMAX_API_KEY
```

## Run

### CLI

```bash
python -m alphaquant AAPL
python -m alphaquant AAPL --format markdown
python -m alphaquant AAPL --pretty --output report.json
```

### FastAPI

```bash
uvicorn alphaquant.main:app --reload
```

Then:
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

Docs at http://localhost:8000/docs

## Test

```bash
python -m tests.smoke
```

## Architecture

8 CrewAI Agents orchestrated via Flow:
1. CompanyResolver → 2. MarketAnalyst (parallel with 3, 4) → 3. NewsAnalyst → 4. FinancialAnalyst → 5. CompetitorAnalyst → 6. RiskAnalyst → 7. ValuationAnalyst → 8. ReportWriter

Data sources (fallback chain): Yahoo Finance → Alpha Vantage → Finnhub → SEC EDGAR → NewsAPI.

LLM: MiniMax-M3 via LiteLLM.