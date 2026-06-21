"""Tests for optional InvestmentReport.confidence field.

Sub-plan for confidence-rubric spec (b0308df). confidence becomes int | None
so the LLM can return null when it cannot justify a number.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphaquant.models.report import InvestmentReport


class TestConfidenceOptional:
    def test_none_accepted(self, stub_report):
        """Confidence can be None (LLM returns null when not justified)."""
        rep = stub_report(confidence=None)
        assert rep.confidence is None

    def test_default_is_none(self, stub_report):
        """When confidence is omitted, default is None (not ValidationError)."""
        rep = stub_report()
        assert rep.confidence is None

    def test_zero_accepted(self, stub_report):
        rep = stub_report(confidence=0)
        assert rep.confidence == 0

    def test_hundred_accepted(self, stub_report):
        rep = stub_report(confidence=100)
        assert rep.confidence == 100

    def test_seventy_accepted(self, stub_report):
        """Regression: numeric confidence values still work."""
        rep = stub_report(confidence=70)
        assert rep.confidence == 70

    def test_negative_rejected(self, stub_report):
        with pytest.raises(ValidationError):
            stub_report(confidence=-1)

    def test_above_100_rejected(self, stub_report):
        with pytest.raises(ValidationError):
            stub_report(confidence=101)
