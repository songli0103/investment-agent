"""Custom exception types for AlphaQuant."""


class AlphaQuantError(Exception):
    """Base exception for all AlphaQuant errors."""


class TickerNotFound(AlphaQuantError):
    """Raised when ticker cannot be resolved by any data source."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        super().__init__(f"Ticker not found: {ticker}")


class InvalidTickerFormat(AlphaQuantError):
    r"""Raised when ticker format is invalid (not matching [A-Z]{1,5}(\.[A-Z])?)."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        super().__init__(
            f"Invalid ticker format: {ticker!r}. Expected like 'AAPL' or 'BRK.B'."
        )


class AllDataSourcesDown(AlphaQuantError):
    """Raised when all data sources for a query fail."""


class PartialDataFailure(AlphaQuantError):
    """Non-fatal: some data sources failed but we have partial results."""

    def __init__(self, message: str, missing_fields: list[str] | None = None):
        self.missing_fields = missing_fields or []
        super().__init__(message)


class ReportGenerationError(AlphaQuantError):
    """Raised when the final report synthesis step fails.

    Per spec §3.2, this maps to HTTP 500 INTERNAL_ERROR.
    """