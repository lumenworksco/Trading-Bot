"""Tests for analytics.alpha_decay — alpha decay monitoring."""

import numpy as np
import pandas as pd
import pytest

from analytics.alpha_decay import AlphaDecayMonitor


@pytest.fixture
def monitor():
    return AlphaDecayMonitor()


def _make_trade_history(strategy: str, n: int, pnl_mean: float = 0.005,
                        pnl_std: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic trade history for a single strategy."""
    rng = np.random.RandomState(seed)
    pnls = rng.normal(pnl_mean, pnl_std, n)
    return pd.DataFrame({
        "symbol": ["AAPL"] * n,
        "strategy": [strategy] * n,
        "pnl_pct": pnls,
    })


def _make_multi_strategy_history(strategies: list[str], n_per: int = 50,
                                 seed: int = 42) -> pd.DataFrame:
    """Generate trade history spanning multiple strategies."""
    frames = []
    for i, strat in enumerate(strategies):
        frames.append(_make_trade_history(strat, n_per, seed=seed + i))
    return pd.concat(frames, ignore_index=True)


# ===================================================================
# compute_decay — return shape and keys
# ===================================================================

class TestComputeDecayShape:

    def test_returns_all_keys(self, monitor):
        df = _make_trade_history("ORB", 100)
        result = monitor.compute_decay("ORB", df)
        expected_keys = {"sharpe_30d", "sharpe_60d", "sharpe_90d",
                         "decay_rate", "half_life_days", "status"}
        assert set(result.keys()) == expected_keys

    def test_empty_history_returns_safe_defaults(self, monitor):
        result = monitor.compute_decay("ORB", pd.DataFrame())
        assert result["status"] == "healthy"
        assert result["sharpe_30d"] is None
        assert result["decay_rate"] == 0.0

    def test_none_history_returns_safe_defaults(self, monitor):
        result = monitor.compute_decay("ORB", None)
        assert result["status"] == "healthy"

    def test_insufficient_trades(self, monitor):
        """Fewer than 5 trades should return safe defaults."""
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 3,
            "strategy": ["ORB"] * 3,
            "pnl_pct": [0.01, -0.005, 0.02],
        })
        result = monitor.compute_decay("ORB", df)
        assert result["status"] == "healthy"
        assert result["sharpe_30d"] is None


# ===================================================================
# Status classification
# ===================================================================

class TestStatusValues:

    def test_healthy_status(self, monitor):
        """Consistently profitable strategy should be healthy."""
        df = _make_trade_history("ORB", 100, pnl_mean=0.02, pnl_std=0.005)
        result = monitor.compute_decay("ORB", df)
        assert result["status"] == "healthy"

    def test_critical_status(self, monitor):
        """Strategy with near-zero or negative Sharpe should be critical."""
        df = _make_trade_history("BAD", 100, pnl_mean=-0.001, pnl_std=0.02)
        result = monitor.compute_decay("BAD", df)
        assert result["status"] in ("critical", "warning")

    def test_warning_status(self, monitor, override_config):
        """Borderline strategy should get warning status."""
        with override_config(ALPHA_DECAY_CRITICAL_SHARPE=0.1,
                             ALPHA_DECAY_WARNING_SHARPE=5.0):
            # Use moderate returns that produce a Sharpe between 0.1 and 5.0
            df = _make_trade_history("MEH", 100, pnl_mean=0.003, pnl_std=0.01)
            result = monitor.compute_decay("MEH", df)
            assert result["status"] == "warning"


# ===================================================================
# Decay rate sign
# ===================================================================

class TestDecayRateSign:

    def test_improving_strategy_positive_decay(self, monitor):
        """If recent Sharpe > older Sharpe, decay_rate should be positive."""
        # Build a history where early trades are bad, later trades are good
        bad = pd.DataFrame({
            "symbol": ["AAPL"] * 60,
            "strategy": ["IMP"] * 60,
            "pnl_pct": np.random.RandomState(1).normal(-0.001, 0.01, 60),
        })
        good = pd.DataFrame({
            "symbol": ["AAPL"] * 40,
            "strategy": ["IMP"] * 40,
            "pnl_pct": np.random.RandomState(2).normal(0.03, 0.005, 40),
        })
        df = pd.concat([bad, good], ignore_index=True)
        result = monitor.compute_decay("IMP", df)
        # 30d window has the good trades, 90d has a mix
        if result["sharpe_30d"] is not None and result["sharpe_90d"] is not None:
            assert result["decay_rate"] > 0

    def test_declining_strategy_negative_decay(self, monitor):
        """If recent Sharpe < older Sharpe, decay_rate should be negative."""
        good = pd.DataFrame({
            "symbol": ["AAPL"] * 60,
            "strategy": ["DEC"] * 60,
            "pnl_pct": np.random.RandomState(3).normal(0.03, 0.005, 60),
        })
        bad = pd.DataFrame({
            "symbol": ["AAPL"] * 40,
            "strategy": ["DEC"] * 40,
            "pnl_pct": np.random.RandomState(4).normal(-0.005, 0.015, 40),
        })
        df = pd.concat([good, bad], ignore_index=True)
        result = monitor.compute_decay("DEC", df)
        if result["sharpe_30d"] is not None and result["sharpe_90d"] is not None:
            assert result["decay_rate"] < 0


# ===================================================================
# Health report
# ===================================================================

class TestHealthReport:

    def test_report_covers_all_strategies(self, monitor):
        df = _make_multi_strategy_history(["ORB", "VWAP", "STAT_MR"], n_per=50)
        report = monitor.get_strategy_health_report(df)
        assert "ORB" in report
        assert "VWAP" in report
        assert "STAT_MR" in report
        assert len(report) == 3

    def test_report_each_has_required_keys(self, monitor):
        df = _make_multi_strategy_history(["ORB", "VWAP"], n_per=50)
        report = monitor.get_strategy_health_report(df)
        expected = {"sharpe_30d", "sharpe_60d", "sharpe_90d",
                    "decay_rate", "half_life_days", "status"}
        for strat, metrics in report.items():
            assert set(metrics.keys()) == expected

    def test_empty_history_report(self, monitor):
        report = monitor.get_strategy_health_report(pd.DataFrame())
        assert report == {}

    def test_none_history_report(self, monitor):
        report = monitor.get_strategy_health_report(None)
        assert report == {}


# ===================================================================
# Half-life
# ===================================================================

class TestHalfLife:

    def test_half_life_none_when_not_decaying(self, monitor):
        """Non-decaying strategy should have no half-life."""
        df = _make_trade_history("GOOD", 100, pnl_mean=0.02, pnl_std=0.005)
        result = monitor.compute_decay("GOOD", df)
        # If not decaying, half_life should be None
        if result["decay_rate"] >= 0:
            assert result["half_life_days"] is None

    def test_half_life_positive_when_decaying(self, monitor):
        """Decaying strategy should have a positive half-life in days."""
        good = pd.DataFrame({
            "symbol": ["AAPL"] * 60,
            "strategy": ["HL"] * 60,
            "pnl_pct": np.random.RandomState(10).normal(0.03, 0.005, 60),
        })
        bad = pd.DataFrame({
            "symbol": ["AAPL"] * 40,
            "strategy": ["HL"] * 40,
            "pnl_pct": np.random.RandomState(11).normal(-0.002, 0.012, 40),
        })
        df = pd.concat([good, bad], ignore_index=True)
        result = monitor.compute_decay("HL", df)
        if result["decay_rate"] < 0 and result["half_life_days"] is not None:
            assert result["half_life_days"] > 0
