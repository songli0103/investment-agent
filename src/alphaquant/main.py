"""AlphaQuant FastAPI app.

The shared analysis core lives in ``alphaquant.core`` to avoid a circular
import with the route module (which also needs to call the core).
"""
from __future__ import annotations

from fastapi import FastAPI

from alphaquant.interfaces.api.routes import router
from alphaquant.observability import configure_logging, get_logger

# Configure structured logging once at import time. Idempotent under
# repeated calls because structlog rewires its global processors cleanly.
configure_logging()
log = get_logger("alphaquant.main")

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
