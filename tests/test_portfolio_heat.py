"""Tests for V8 Portfolio Heat & Correlation Clustering."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")


def _make_trade(symbol="AAPL", entry_price=150.0, qty=10, strategy="STAT_MR",
                side="buy"):
    from risk import TradeRecord
    return TradeRecord(
        symbol=symbol, strategy=strategy, side=side,
        entry_price=entry_price, qty=qty, take_profit=155.0,
        stop_loss=148.0, pnl=0.0, exit_price=None, exit_reason="",
        status="open", hold_type="day", pair_id="",
        entry_time=datetime(2026, 3, 13, 10, 5, tzinfo=ET),
        partial_exits=0, highest_price_seen=entry_price, entry_atr=1.0,
    )


class TestPortfolioHeatTracker:

    def _make_tracker(self):
        from risk.portfolio_heat import PortfolioHeatTracker
        return PortfolioHeatTracker()

    def test_empty_portfolio_zero_heat(self, override_config):
        tracker = self._make_tracker()
        with override_config(PORTFOLIO_HEAT_ENABLED=True):
            heat = tracker.compute_heat({}, 100000)
            assert heat == 0.0

    def test_single_position_heat(self, override_config):
        tracker = self._make_tracker()
        trades = {"AAPL": _make_trade("AAPL", 150.0, 10)}
        with override_config(PORTFOLIO_HEAT_ENABLED=True):
            heat = tracker.compute_heat(trades, 100000)
            # 150*10/100000 = 0.015 * sqrt(1) = 0.015
            assert abs(heat - 0.015) < 0.001

    def test_check_new_trade_under_limit(self, override_config):
        tracker = self._make_tracker()
        with override_config(PORTFOLIO_HEAT_ENABLED=True, PORTFOLIO_HEAT_MAX=0.60,
                           CLUSTER_MAX_HEAT=0.20):
            allowed, reason = tracker.check_new_trade("MSFT", 1000, {}, 100000)
            assert allowed is True

    def test_check_new_trade_over_total_heat(self, override_config):
        tracker = self._make_tracker()
        # Create trades that take up 58% of portfolio
        trades = {"AAPL": _make_trade("AAPL", 100.0, 580)}  # $58,000 of $100,000
        with override_config(PORTFOLIO_HEAT_ENABLED=True, PORTFOLIO_HEAT_MAX=0.60,
                           CLUSTER_MAX_HEAT=0.20):
            allowed, reason = tracker.check_new_trade("MSFT", 5000, trades, 100000)
            assert allowed is False
            assert "portfolio_heat" in reason

    def test_disabled_always_allows(self, override_config):
        tracker = self._make_tracker()
        with override_config(PORTFOLIO_HEAT_ENABLED=False):
            allowed, reason = tracker.check_new_trade("MSFT", 90000, {}, 100000)
            assert allowed is True

    def test_cluster_heat_limit(self, override_config):
        tracker = self._make_tracker()
        # Manually set clusters
        tracker._clusters = {0: ["AAPL", "MSFT", "GOOGL"]}

        trades = {
            "AAPL": _make_trade("AAPL", 100.0, 100),   # $10,000
            "MSFT": _make_trade("MSFT", 100.0, 100),    # $10,000
        }
        # Cluster has $20k/100k = 20%, adding more would exceed
        with override_config(PORTFOLIO_HEAT_ENABLED=True, PORTFOLIO_HEAT_MAX=0.60,
                           CLUSTER_MAX_HEAT=0.20):
            allowed, reason = tracker.check_new_trade("GOOGL", 5000, trades, 100000)
            assert allowed is False
            assert "cluster_heat" in reason

    def test_reset_daily(self):
        tracker = self._make_tracker()
        tracker._current_heat = 0.5
        tracker.reset_daily()
        assert tracker.current_heat == 0.0

    def test_heat_disabled_returns_zero(self, override_config):
        tracker = self._make_tracker()
        trades = {"AAPL": _make_trade("AAPL", 150.0, 100)}
        with override_config(PORTFOLIO_HEAT_ENABLED=False):
            heat = tracker.compute_heat(trades, 100000)
            assert heat == 0.0
