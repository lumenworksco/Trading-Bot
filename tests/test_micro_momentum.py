"""Tests for IntradayMicroMomentum strategy."""

from datetime import datetime, timedelta
from dataclasses import dataclass
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")


def _make_spy_bars(num_bars=25, last_volume=5_000_000, avg_volume=1_000_000,
                   last_close=450.0, prev_close=449.0):
    """Build a DataFrame mimicking SPY 1-min bars for event detection."""
    volumes = [avg_volume] * (num_bars - 1) + [last_volume]
    closes = [prev_close] * (num_bars - 1) + [last_close]
    idx = pd.date_range(
        end=datetime(2026, 3, 13, 10, 30, tzinfo=ET),
        periods=num_bars, freq="min",
    )
    return pd.DataFrame({"close": closes, "volume": volumes}, index=idx)


def _make_stock_bars(price=100.0, num_bars=5):
    """Build a small DataFrame for stock price lookups."""
    idx = pd.date_range(
        end=datetime(2026, 3, 13, 10, 30, tzinfo=ET),
        periods=num_bars, freq="min",
    )
    return pd.DataFrame({
        "close": [price] * num_bars,
        "volume": [500_000] * num_bars,
    }, index=idx)


@dataclass
class MockTrade:
    symbol: str
    strategy: str
    entry_time: datetime
    side: str = "buy"


class TestDetectEvent:
    """Tests for detect_event()."""

    @patch("strategies.micro_momentum.get_intraday_bars")
    def test_detect_event_spy_volume_spike(self, mock_bars):
        """High volume + price move triggers event detection."""
        from strategies.micro_momentum import IntradayMicroMomentum

        # Volume spike: 5M vs 1M avg = 5x (> 3.0 threshold)
        # Price move: (450 - 449) / 449 = 0.22% (> 0.15%)
        mock_bars.return_value = _make_spy_bars(
            last_volume=5_000_000, avg_volume=1_000_000,
            last_close=450.0, prev_close=449.0,
        )

        strat = IntradayMicroMomentum()
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        result = strat.detect_event(now)

        assert result is True
        assert strat._event_active is True
        assert strat._event_direction == "up"
        assert strat._event_time == now

    @patch("strategies.micro_momentum.get_intraday_bars")
    def test_detect_event_no_spike(self, mock_bars):
        """Normal volume does not trigger event."""
        from strategies.micro_momentum import IntradayMicroMomentum

        # Volume ratio ~1.0 (no spike)
        mock_bars.return_value = _make_spy_bars(
            last_volume=1_000_000, avg_volume=1_000_000,
            last_close=450.0, prev_close=449.9,
        )

        strat = IntradayMicroMomentum()
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        result = strat.detect_event(now)

        assert result is False
        assert strat._event_active is False

    @patch("strategies.micro_momentum.get_intraday_bars")
    def test_detect_event_down_direction(self, mock_bars):
        """Negative price move sets direction to 'down'."""
        from strategies.micro_momentum import IntradayMicroMomentum

        mock_bars.return_value = _make_spy_bars(
            last_volume=5_000_000, avg_volume=1_000_000,
            last_close=448.0, prev_close=449.0,
        )

        strat = IntradayMicroMomentum()
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        result = strat.detect_event(now)

        assert result is True
        assert strat._event_direction == "down"

    @patch("strategies.micro_momentum.get_intraday_bars")
    def test_detect_event_none_bars(self, mock_bars):
        """Returns False when bars are None."""
        from strategies.micro_momentum import IntradayMicroMomentum

        mock_bars.return_value = None
        strat = IntradayMicroMomentum()
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)

        assert strat.detect_event(now) is False

    @patch("strategies.micro_momentum.get_intraday_bars")
    def test_event_cooldown(self, mock_bars):
        """Within 15-min cooldown, detect_event returns current state."""
        from strategies.micro_momentum import IntradayMicroMomentum

        mock_bars.return_value = _make_spy_bars(
            last_volume=5_000_000, avg_volume=1_000_000,
            last_close=450.0, prev_close=449.0,
        )

        strat = IntradayMicroMomentum()
        t1 = datetime(2026, 3, 13, 10, 0, tzinfo=ET)
        strat.detect_event(t1)
        assert strat._event_active is True

        # Try again 5 min later — within cooldown
        t2 = t1 + timedelta(minutes=5)
        result = strat.detect_event(t2)

        # Should return current _event_active without re-checking bars
        assert result is True
        # get_intraday_bars should only have been called once (for t1)
        assert mock_bars.call_count == 1


class TestScan:
    """Tests for scan()."""

    @patch("strategies.micro_momentum.get_intraday_bars")
    def test_scan_generates_signals_on_event(self, mock_bars):
        """Active up-event produces buy signals for top-beta stocks."""
        from strategies.micro_momentum import IntradayMicroMomentum

        mock_bars.return_value = _make_stock_bars(price=100.0)

        strat = IntradayMicroMomentum()
        strat._event_active = True
        strat._event_direction = "up"
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        strat._event_time = now - timedelta(minutes=1)

        with patch.dict("config.__dict__", {
            "MICRO_MAX_TRADES_PER_EVENT": 3,
            "MICRO_MAX_DAILY_GAIN_DISABLE": 0.015,
            "MICRO_TOP_BETA_STOCKS": 5,
            "MICRO_STOP_PCT": 0.003,
            "MICRO_TARGET_PCT": 0.006,
            "ALLOW_SHORT": False,
            "LEVERAGED_ETFS": set(),
            "NO_SHORT_SYMBOLS": set(),
        }):
            signals = strat.scan(now)

        assert len(signals) == 3  # capped at MAX_TRADES_PER_EVENT
        for sig in signals:
            assert sig.strategy == "MICRO_MOM"
            assert sig.side == "buy"
            assert sig.take_profit > sig.entry_price
            assert sig.stop_loss < sig.entry_price

    @patch("strategies.micro_momentum.get_intraday_bars")
    def test_scan_max_trades_per_event(self, mock_bars):
        """Only MICRO_MAX_TRADES_PER_EVENT signals are generated."""
        from strategies.micro_momentum import IntradayMicroMomentum

        mock_bars.return_value = _make_stock_bars(price=50.0)

        strat = IntradayMicroMomentum()
        strat._event_active = True
        strat._event_direction = "up"
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        strat._event_time = now - timedelta(minutes=1)

        with patch.dict("config.__dict__", {
            "MICRO_MAX_TRADES_PER_EVENT": 2,
            "MICRO_MAX_DAILY_GAIN_DISABLE": 0.015,
            "MICRO_TOP_BETA_STOCKS": 10,
            "MICRO_STOP_PCT": 0.003,
            "MICRO_TARGET_PCT": 0.006,
            "ALLOW_SHORT": False,
            "LEVERAGED_ETFS": set(),
            "NO_SHORT_SYMBOLS": set(),
        }):
            signals = strat.scan(now)

        assert len(signals) == 2

    def test_scan_disabled_when_pnl_high(self):
        """No signals when day P&L already exceeds disable threshold."""
        from strategies.micro_momentum import IntradayMicroMomentum

        strat = IntradayMicroMomentum()
        strat._event_active = True
        strat._event_direction = "up"
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        strat._event_time = now - timedelta(minutes=1)

        with patch.dict("config.__dict__", {
            "MICRO_MAX_DAILY_GAIN_DISABLE": 0.015,
        }):
            signals = strat.scan(now, day_pnl_pct=0.02)

        assert signals == []

    def test_scan_no_signal_when_no_event(self):
        """No signals when no event is active."""
        from strategies.micro_momentum import IntradayMicroMomentum

        strat = IntradayMicroMomentum()
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        signals = strat.scan(now)
        assert signals == []

    @patch("strategies.micro_momentum.get_intraday_bars")
    def test_scan_event_window_expired(self, mock_bars):
        """No signals when event is older than 5 minutes."""
        from strategies.micro_momentum import IntradayMicroMomentum

        strat = IntradayMicroMomentum()
        strat._event_active = True
        strat._event_direction = "up"
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        strat._event_time = now - timedelta(minutes=6)

        with patch.dict("config.__dict__", {
            "MICRO_MAX_TRADES_PER_EVENT": 3,
            "MICRO_MAX_DAILY_GAIN_DISABLE": 0.015,
        }):
            signals = strat.scan(now)

        assert signals == []
        assert strat._event_active is False


class TestCheckExits:
    """Tests for check_exits()."""

    def test_check_exits_time_stop(self):
        """Trade held longer than 8 min gets time-stopped."""
        from strategies.micro_momentum import IntradayMicroMomentum

        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        trade = MockTrade(
            symbol="NVDA",
            strategy="MICRO_MOM",
            entry_time=now - timedelta(minutes=9),
        )

        strat = IntradayMicroMomentum()
        with patch.dict("config.__dict__", {"MICRO_MAX_HOLD_MINUTES": 8}):
            exits = strat.check_exits({"NVDA": trade}, now)

        assert len(exits) == 1
        assert exits[0]["symbol"] == "NVDA"
        assert exits[0]["action"] == "full"
        assert "time stop" in exits[0]["reason"]

    def test_check_exits_no_exit_before_time(self):
        """Trade within hold window is not exited."""
        from strategies.micro_momentum import IntradayMicroMomentum

        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        trade = MockTrade(
            symbol="NVDA",
            strategy="MICRO_MOM",
            entry_time=now - timedelta(minutes=5),
        )

        strat = IntradayMicroMomentum()
        with patch.dict("config.__dict__", {"MICRO_MAX_HOLD_MINUTES": 8}):
            exits = strat.check_exits({"NVDA": trade}, now)

        assert exits == []

    def test_check_exits_ignores_other_strategies(self):
        """Non-MICRO_MOM trades are skipped."""
        from strategies.micro_momentum import IntradayMicroMomentum

        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        trade = MockTrade(
            symbol="AAPL",
            strategy="ORB",
            entry_time=now - timedelta(minutes=60),
        )

        strat = IntradayMicroMomentum()
        exits = strat.check_exits({"AAPL": trade}, now)
        assert exits == []


class TestResetDaily:
    """Tests for reset_daily()."""

    def test_reset_daily(self):
        """All internal state is cleared on daily reset."""
        from strategies.micro_momentum import IntradayMicroMomentum

        strat = IntradayMicroMomentum()
        strat._event_active = True
        strat._event_direction = "up"
        strat._event_time = datetime(2026, 3, 13, 10, 0, tzinfo=ET)
        strat._daily_trade_count = 5
        strat._trades_this_event = 3
        strat._triggered_symbols = {"NVDA", "TSLA"}

        strat.reset_daily()

        assert strat._event_active is False
        assert strat._event_direction == ""
        assert strat._event_time is None
        assert strat._daily_trade_count == 0
        assert strat._trades_this_event == 0
        assert strat._triggered_symbols == set()
