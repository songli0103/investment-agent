"""Financial statements models."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field


class IncomeStatement(BaseModel):
    period: Literal["TTM", "FY", "Q1", "Q2", "Q3", "Q4"]
    fiscal_year: int = Field(..., ge=1900, le=2100)
    revenue: Decimal = Field(..., ge=0)
    cogs: Decimal | None = None
    gross_profit: Decimal | None = None
    operating_income: Decimal | None = None
    net_income: Decimal
    eps: Decimal | None = None


class BalanceSheet(BaseModel):
    period: Literal["Q1", "Q2", "Q3", "Q4", "FY"]
    fiscal_year: int = Field(..., ge=1900, le=2100)
    total_assets: Decimal = Field(..., ge=0)
    total_liabilities: Decimal = Field(..., ge=0)
    total_equity: Decimal
    cash_and_equivalents: Decimal | None = None
    short_term_debt: Decimal | None = None
    long_term_debt: Decimal | None = None


class CashFlowStatement(BaseModel):
    period: Literal["TTM", "FY"]
    fiscal_year: int = Field(..., ge=1900, le=2100)
    operating_cash_flow: Decimal
    investing_cash_flow: Decimal | None = None
    financing_cash_flow: Decimal | None = None
    free_cash_flow: Decimal | None = None
    capex: Decimal | None = None


class FinancialStatements(BaseModel):
    ticker: str
    income_statements: list[IncomeStatement] = Field(default_factory=list)
    balance_sheets: list[BalanceSheet] = Field(default_factory=list)
    cash_flows: list[CashFlowStatement] = Field(default_factory=list)
    source: str = "unavailable"
