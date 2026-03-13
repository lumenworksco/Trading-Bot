"""Tests for Beta Neutralizer."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


@dataclass
class MockTrade:
    symbol: str
    side: str
    qty: int
    entry_price: float


class TestBetaNeutralizer:
    def test_empty_portfolio_zero_beta(self):
        from risk.beta_neutralizer import BetaNeutralizer
        bn = BetaNeutralizer()
        beta = bn.compute_portfolio_beta({}, {})
        assert beta == 0.0

    def test_long_position_positive_beta(self):
        from risk.beta_neutralizer import BetaNeutralizer
        bn = BetaNeutralizer()
        trades = {'NVDA': MockTrade('NVDA', 'buy', 10, 800.0)}
        prices = {'NVDA': 800.0}
        beta = bn.compute_portfolio_beta(trades, prices)
        assert beta > 0  # NVDA has beta > 1

    def test_short_position_negative_beta(self):
        from risk.beta_neutralizer import BetaNeutralizer
        bn = BetaNeutralizer()
        trades = {'NVDA': MockTrade('NVDA', 'sell', 10, 800.0)}
        prices = {'NVDA': 800.0}
        beta = bn.compute_portfolio_beta(trades, prices)
        assert beta < 0

    def test_needs_hedge_above_threshold(self):
        with patch.dict('config.__dict__', {'BETA_MAX_ABS': 0.3}):
            from risk.beta_neutralizer import BetaNeutralizer
            bn = BetaNeutralizer()
            bn._portfolio_beta = 0.5
            assert bn.needs_hedge()

    def test_no_hedge_within_band(self):
        with patch.dict('config.__dict__', {'BETA_MAX_ABS': 0.3}):
            from risk.beta_neutralizer import BetaNeutralizer
            bn = BetaNeutralizer()
            bn._portfolio_beta = 0.1
            assert not bn.needs_hedge()

    def test_hedge_signal_long_when_net_short(self):
        with patch.dict('config.__dict__', {
            'BETA_MAX_ABS': 0.3,
            'ALLOW_SHORT': True,
        }):
            from risk.beta_neutralizer import BetaNeutralizer
            bn = BetaNeutralizer()
            bn._portfolio_beta = -0.5
            signal = bn.compute_hedge_signal(100000, 500.0)
            assert signal is not None
            assert signal.side == "buy"
            assert signal.symbol == "SPY"

    def test_hedge_signal_short_when_net_long(self):
        with patch.dict('config.__dict__', {
            'BETA_MAX_ABS': 0.3,
            'ALLOW_SHORT': True,
        }):
            from risk.beta_neutralizer import BetaNeutralizer
            bn = BetaNeutralizer()
            bn._portfolio_beta = 0.5
            signal = bn.compute_hedge_signal(100000, 500.0)
            assert signal is not None
            assert signal.side == "sell"
            assert signal.symbol == "SPY"

    def test_should_skip_opening(self):
        from risk.beta_neutralizer import BetaNeutralizer
        bn = BetaNeutralizer()
        # 9:35 AM should be skipped (within first 15 min)
        opening = datetime(2026, 3, 13, 9, 35, tzinfo=__import__('zoneinfo').ZoneInfo("America/New_York"))
        assert bn.should_skip(opening)

        # 10:00 AM should NOT be skipped
        later = datetime(2026, 3, 13, 10, 0, tzinfo=__import__('zoneinfo').ZoneInfo("America/New_York"))
        assert not bn.should_skip(later)

    def test_reset_daily(self):
        from risk.beta_neutralizer import BetaNeutralizer
        bn = BetaNeutralizer()
        bn._portfolio_beta = 0.5
        bn.reset_daily()
        assert bn.portfolio_beta == 0.0
