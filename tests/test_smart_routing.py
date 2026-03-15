"""Tests for smart_routing.py — SmartOrderRouter and FillMonitor."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import config
from smart_routing import SmartOrderRouter, FillMonitor, OrderParams
from tests.conftest import _make_signal

ET = ZoneInfo("America/New_York")


# ===================================================================
# SmartOrderRouter tests
# ===================================================================

class TestSmartOrderRouter:
    """Test the SmartOrderRouter decision tree."""

    def setup_method(self):
        self.router = SmartOrderRouter()

    # --- Rule 2: Time-sensitive strategies -> IOC ---

    def test_orb_returns_ioc(self):
        """ORB is time-sensitive and should use IOC market order."""
        signal = _make_signal(strategy="ORB")
        params = self.router.route(signal, qty=10)
        assert params.order_type == "ioc"
        assert params.urgency == "high"
        assert params.limit_price is None
        assert params.use_twap is False

    def test_micro_mom_returns_ioc(self):
        """MICRO_MOM is time-sensitive and should use IOC market order."""
        signal = _make_signal(strategy="MICRO_MOM", entry_price=50.0,
                              take_profit=53.0, stop_loss=48.5)
        params = self.router.route(signal, qty=10)
        assert params.order_type == "ioc"
        assert params.urgency == "high"

    # --- Rule 4: Mean-reversion strategies -> limit ---

    def test_stat_mr_returns_limit(self):
        """STAT_MR is mean-reversion and should use limit order."""
        signal = _make_signal(strategy="STAT_MR")
        params = self.router.route(signal, qty=10)
        assert params.order_type == "limit"
        assert params.urgency == "low"
        assert params.limit_price == signal.entry_price

    def test_vwap_returns_limit(self):
        """VWAP is mean-reversion and should use limit order."""
        signal = _make_signal(strategy="VWAP")
        params = self.router.route(signal, qty=10)
        assert params.order_type == "limit"
        assert params.urgency == "low"
        assert params.limit_price == signal.entry_price

    def test_kalman_pairs_returns_limit(self):
        """KALMAN_PAIRS is mean-reversion and should use limit order."""
        signal = _make_signal(strategy="KALMAN_PAIRS", pair_id="A-B-001")
        params = self.router.route(signal, qty=10)
        assert params.order_type == "limit"
        assert params.urgency == "low"

    # --- Rule 1: Wide spread -> limit ---

    def test_wide_spread_returns_limit(self):
        """When spread exceeds threshold, should use limit order regardless of strategy."""
        signal = _make_signal(strategy="MICRO_MOM", entry_price=50.0,
                              take_profit=53.0, stop_loss=48.5)
        # Spread of 0.5% > threshold of 0.15%
        params = self.router.route(signal, qty=10, spread_pct=0.005)
        assert params.order_type == "limit"
        assert params.urgency == "medium"

    def test_narrow_spread_does_not_trigger_limit(self):
        """Spread below threshold should not force limit order."""
        signal = _make_signal(strategy="MICRO_MOM", entry_price=50.0,
                              take_profit=53.0, stop_loss=48.5)
        params = self.router.route(signal, qty=10, spread_pct=0.0005)
        # MICRO_MOM should still route to IOC, not limit
        assert params.order_type == "ioc"

    # --- Rule 5: PEAD -> limit with offset ---

    def test_pead_buy_uses_limit_with_positive_offset(self):
        """PEAD buy should use limit at entry + 0.3%."""
        signal = _make_signal(strategy="PEAD", side="buy", entry_price=100.0,
                              take_profit=105.0, stop_loss=97.0)
        params = self.router.route(signal, qty=10)
        assert params.order_type == "limit"
        assert params.urgency == "medium"
        assert params.limit_price == round(100.0 * 1.003, 2)

    def test_pead_sell_uses_limit_with_negative_offset(self):
        """PEAD sell should use limit at entry - 0.3%."""
        signal = _make_signal(strategy="PEAD", side="sell", entry_price=100.0,
                              take_profit=95.0, stop_loss=103.0)
        params = self.router.route(signal, qty=10)
        assert params.order_type == "limit"
        assert params.limit_price == round(100.0 * 0.997, 2)

    # --- Rule 3: Large orders -> TWAP ---

    def test_large_order_uses_twap(self):
        """Order value > 3% of equity should trigger TWAP."""
        # equity=100k, 3% = $3000. 100 shares * $50 = $5000 > $3000
        signal = _make_signal(strategy="MOMENTUM", entry_price=50.0,
                              take_profit=55.0, stop_loss=47.0)
        params = self.router.route(signal, qty=100, equity=100_000.0)
        assert params.use_twap is True
        assert params.twap_slices > 0
        assert params.twap_interval_sec > 0

    def test_small_order_no_twap(self):
        """Order value < 3% of equity should not trigger TWAP."""
        # 10 shares * $50 = $500 < $3000
        signal = _make_signal(strategy="MOMENTUM", entry_price=50.0,
                              take_profit=55.0, stop_loss=47.0)
        params = self.router.route(signal, qty=10, equity=100_000.0)
        assert params.use_twap is False

    # --- Adaptive TWAP slices ---

    def test_adaptive_twap_high_urgency(self):
        """High urgency TWAP: 3 slices, 15 sec."""
        signal = _make_signal(strategy="ORB")
        slices, interval = self.router.compute_adaptive_twap(signal, 100, "high")
        assert slices == 3
        assert interval == 15

    def test_adaptive_twap_medium_urgency(self):
        """Medium urgency TWAP: 5 slices, 30 sec."""
        signal = _make_signal(strategy="PEAD")
        slices, interval = self.router.compute_adaptive_twap(signal, 100, "medium")
        assert slices == 5
        assert interval == 30

    def test_adaptive_twap_low_urgency(self):
        """Low urgency TWAP: 8 slices, 60 sec."""
        signal = _make_signal(strategy="STAT_MR")
        slices, interval = self.router.compute_adaptive_twap(signal, 100, "low")
        assert slices == 8
        assert interval == 60

    # --- Disabled routing ---

    def test_disabled_returns_market(self, override_config):
        """When SMART_ROUTING_ENABLED=False, always return market order."""
        with override_config(SMART_ROUTING_ENABLED=False):
            signal = _make_signal(strategy="STAT_MR")
            params = self.router.route(signal, qty=10)
            assert params.order_type == "market"
            assert params.limit_price is None
            assert params.use_twap is False

    # --- Unknown strategy defaults ---

    def test_unknown_strategy_defaults_to_market(self):
        """Unknown strategies should get a plain market order."""
        signal = _make_signal(strategy="UNKNOWN_STRAT")
        params = self.router.route(signal, qty=10)
        assert params.order_type == "market"
        assert params.urgency == "medium"


# ===================================================================
# FillMonitor tests
# ===================================================================

class TestFillMonitor:
    """Test the FillMonitor order tracking and analytics."""

    def setup_method(self):
        self.monitor = FillMonitor()
        self.now = datetime(2026, 3, 15, 10, 0, tzinfo=ET)

    def test_register_order(self):
        """Registering an order adds it to pending."""
        signal = _make_signal()
        self.monitor.register_order("ord-1", signal, self.now, 10)
        assert self.monitor.pending_count == 1

    def test_check_pending_no_actions_before_timeout(self):
        """No actions should be returned before CHASE_AFTER_SECONDS."""
        signal = _make_signal()
        self.monitor.register_order("ord-1", signal, self.now, 10)
        # Check 30 seconds later (< 60 sec chase threshold)
        actions = self.monitor.check_pending(self.now + timedelta(seconds=30))
        assert actions == []

    def test_chase_action_after_timeout(self):
        """After CHASE_AFTER_SECONDS, a chase action should be returned."""
        signal = _make_signal(side="buy")
        self.monitor.register_order("ord-1", signal, self.now, 10)
        # Check 65 seconds later (> 60 sec, < 120 sec)
        actions = self.monitor.check_pending(self.now + timedelta(seconds=65))
        assert len(actions) == 1
        assert actions[0]["action"] == "chase"
        assert actions[0]["order_id"] == "ord-1"
        assert "new_price" in actions[0]

    def test_convert_market_after_longer_timeout(self):
        """After CHASE_CONVERT_MARKET_AFTER, convert_market action should be returned."""
        signal = _make_signal()
        self.monitor.register_order("ord-1", signal, self.now, 10)
        # Check 130 seconds later (> 120 sec)
        actions = self.monitor.check_pending(self.now + timedelta(seconds=130))
        assert len(actions) == 1
        assert actions[0]["action"] == "convert_market"
        assert actions[0]["order_id"] == "ord-1"

    def test_record_fill_and_slippage_stats(self):
        """Recording fills should produce correct slippage stats."""
        self.monitor.record_fill("ord-1", fill_price=150.50,
                                 expected_price=150.0, strategy="ORB")
        self.monitor.record_fill("ord-2", fill_price=150.75,
                                 expected_price=150.0, strategy="ORB")
        stats = self.monitor.get_slippage_stats()
        assert "ORB" in stats
        # (0.50/150 + 0.75/150) / 2
        expected_avg = ((150.50 - 150.0) / 150.0 + (150.75 - 150.0) / 150.0) / 2
        assert abs(stats["ORB"] - expected_avg) < 1e-8

    def test_record_fill_removes_from_pending(self):
        """Recording a fill should remove the order from pending."""
        signal = _make_signal()
        self.monitor.register_order("ord-1", signal, self.now, 10)
        assert self.monitor.pending_count == 1
        self.monitor.record_fill("ord-1", 150.0, 150.0, "ORB")
        assert self.monitor.pending_count == 0

    def test_slippage_stats_multiple_strategies(self):
        """Slippage stats should be tracked per strategy."""
        self.monitor.record_fill("o1", 150.30, 150.0, "ORB")
        self.monitor.record_fill("o2", 100.10, 100.0, "VWAP")
        stats = self.monitor.get_slippage_stats()
        assert "ORB" in stats
        assert "VWAP" in stats

    def test_slippage_stats_empty(self):
        """Empty monitor should return empty stats."""
        stats = self.monitor.get_slippage_stats()
        assert stats == {}

    def test_remove_order(self):
        """Manually removing an order should work."""
        signal = _make_signal()
        self.monitor.register_order("ord-1", signal, self.now, 10)
        self.monitor.remove_order("ord-1")
        assert self.monitor.pending_count == 0

    def test_check_pending_fail_open(self):
        """check_pending should return empty list on internal error, not raise."""
        # Corrupt pending data to trigger an error
        self.monitor._pending["bad"] = {"signal": None, "submit_time": "not-a-datetime", "qty": 0}
        actions = self.monitor.check_pending(self.now)
        assert actions == []

    def test_chase_price_direction_buy(self):
        """Chase on a buy order should increase the price."""
        signal = _make_signal(side="buy", entry_price=100.0)
        self.monitor.register_order("ord-1", signal, self.now, 10)
        actions = self.monitor.check_pending(self.now + timedelta(seconds=65))
        assert actions[0]["new_price"] > 100.0

    def test_chase_price_direction_sell(self):
        """Chase on a sell order should decrease the price."""
        signal = _make_signal(side="sell", entry_price=100.0,
                              take_profit=95.0, stop_loss=103.0)
        self.monitor.register_order("ord-1", signal, self.now, 10)
        actions = self.monitor.check_pending(self.now + timedelta(seconds=65))
        assert actions[0]["new_price"] < 100.0
