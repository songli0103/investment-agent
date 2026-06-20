"""AlphaQuant FastAPI app.

The shared analysis core lives in ``alphaquant.core`` to avoid a circular
import with the route module (which also needs to call the core).
"""
from __future__ import annotations

from fastapi import FastAPI

from alphaquant.api.routes import router

VERSION = "1.0.0"

app = FastAPI(
    title="AlphaQuant",
    description="AI Investment Research Analyst",
    version=VERSION,
)
app.include_router(router, prefix="/api/v1")


# Re-export the shared core so existing imports of
# ``alphaquant.main.run_analysis`` / ``alphaquant.main.run_analysis_async``
# continue to work.
from alphaquant.core import run_analysis, run_analysis_async  # noqa: E402,F401
