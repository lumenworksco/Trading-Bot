"""Tests for V8 Unified Risk Decision Pipeline."""

import pytest


class TestRiskDecision:

    def test_approve(self):
        from risk.decision import RiskDecision, RiskVerdict
        d = RiskDecision.approve(100)
        assert d.verdict == RiskVerdict.APPROVED
        assert d.approved_qty == 100
        assert d.is_approved

    def test_reject(self):
        from risk.decision import RiskDecision, RiskVerdict
        d = RiskDecision.approve(100)
        d.reject("circuit_breaker")
        assert d.verdict == RiskVerdict.REJECTED
        assert d.approved_qty == 0
        assert not d.is_approved
        assert "circuit_breaker" in d.reasons

    def test_modify(self):
        from risk.decision import RiskDecision, RiskVerdict
        d = RiskDecision.approve(100)
        d.modify(50, "pnl_lock_reduction")
        assert d.verdict == RiskVerdict.MODIFIED
        assert d.approved_qty == 50
        assert d.is_approved
        assert d.size_adjustment_factor == 0.5

    def test_multiple_reasons(self):
        from risk.decision import RiskDecision
        d = RiskDecision.approve(100)
        d.add_reason("vix_scaling")
        d.add_reason("bearish_regime")
        assert len(d.reasons) == 2
        assert "vix_scaling; bearish_regime" == d.reasons_str

    def test_reject_sets_qty_zero(self):
        from risk.decision import RiskDecision
        d = RiskDecision.approve(100)
        d.reject("max_positions")
        assert d.approved_qty == 0

    def test_modify_from_zero_original(self):
        from risk.decision import RiskDecision
        d = RiskDecision.approve(0)
        d.modify(0, "no_change")
        assert d.size_adjustment_factor == 0
