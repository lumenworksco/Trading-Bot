"""Tests for ab_testing.py — PaperABTest, ABTestResult."""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ET = ZoneInfo("America/New_York")

# Use a temporary test directory for A/B test files
TEST_AB_DIR = Path(__file__).resolve().parent / "_test_ab_tmp"


@pytest.fixture(autouse=True)
def clean_ab_dir():
    """Create and clean up temporary A/B test directory."""
    TEST_AB_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_AB_DIR.exists():
        shutil.rmtree(TEST_AB_DIR)


@pytest.fixture
def ab_test():
    """Return a PaperABTest with the test directory."""
    import ab_testing
    ab_testing.AB_TESTS_DIR = TEST_AB_DIR
    return ab_testing.PaperABTest()


# ============================================================
# ABTestResult dataclass tests
# ============================================================

class TestABTestResult:
    """Test ABTestResult dataclass."""

    def test_default_creation(self):
        from ab_testing import ABTestResult
        r = ABTestResult()
        assert r.name == ""
        assert r.status == "pending"
        assert r.significant is False
        assert r.config_a_stats["signal_count"] == 0
        assert r.config_b_stats["total_pnl"] == 0.0

    def test_creation_with_values(self):
        from ab_testing import ABTestResult
        r = ABTestResult(
            name="test_orb_params",
            start_date="2026-03-01",
            end_date="2026-03-07",
            duration_days=7,
            status="completed",
            significant=True,
        )
        assert r.name == "test_orb_params"
        assert r.duration_days == 7
        assert r.significant is True

    def test_independent_defaults(self):
        """Each instance gets its own default dicts (no shared mutable default)."""
        from ab_testing import ABTestResult
        r1 = ABTestResult(name="a")
        r2 = ABTestResult(name="b")
        r1.config_a_stats["signal_count"] = 42
        assert r2.config_a_stats["signal_count"] == 0


# ============================================================
# PaperABTest setup tests
# ============================================================

class TestPaperABTestSetup:
    """Test A/B test setup."""

    def test_setup_creates_file(self, ab_test):
        result = ab_test.setup_test("test1", {"ORB_ENABLED": False}, 7)
        assert result is True
        test_file = TEST_AB_DIR / "test1.json"
        assert test_file.exists()
        data = json.loads(test_file.read_text())
        assert data["name"] == "test1"
        assert data["status"] == "active"
        assert data["duration_days"] == 7
        assert data["config_b_overrides"] == {"ORB_ENABLED": False}

    def test_setup_disabled(self, ab_test):
        import config as cfg
        original = getattr(cfg, "AB_TESTING_ENABLED", True)
        try:
            cfg.AB_TESTING_ENABLED = False
            result = ab_test.setup_test("test2", {"ORB_ENABLED": False})
            assert result is False
        finally:
            cfg.AB_TESTING_ENABLED = original

    def test_setup_fail_open(self, ab_test):
        """Setup doesn't crash on I/O errors."""
        import ab_testing
        ab_testing.AB_TESTS_DIR = Path("/nonexistent/path/that/doesnt/exist")
        test = ab_testing.PaperABTest.__new__(ab_testing.PaperABTest)
        result = test.setup_test("test3", {"ORB_ENABLED": False})
        assert result is False
        # Restore
        ab_testing.AB_TESTS_DIR = TEST_AB_DIR


# ============================================================
# PaperABTest results tests
# ============================================================

class TestPaperABTestResults:
    """Test A/B test results computation."""

    def test_get_results_not_found(self, ab_test):
        result = ab_test.get_results("nonexistent")
        assert result.status == "not_found"

    def test_get_results_active_test(self, ab_test):
        ab_test.setup_test("active_test", {"ORB_ENABLED": False}, 7)
        # Add some shadow trades
        ab_test.record_shadow_trade("active_test", {"pnl": 10.0})
        ab_test.record_shadow_trade("active_test", {"pnl": -5.0})
        ab_test.record_shadow_trade("active_test", {"pnl": 20.0})

        with patch.object(ab_test, "_get_production_trades", return_value=[
            {"pnl": 15.0}, {"pnl": -3.0}, {"pnl": 8.0}
        ]):
            result = ab_test.get_results("active_test")

        assert result.status == "active"
        assert result.config_a_stats["signal_count"] == 3
        assert result.config_b_stats["signal_count"] == 3
        assert result.config_a_stats["total_pnl"] == pytest.approx(20.0)
        assert result.config_b_stats["total_pnl"] == pytest.approx(25.0)

    def test_get_results_fail_open(self, ab_test):
        """Results computation doesn't crash on errors."""
        ab_test.setup_test("error_test", {}, 7)
        with patch.object(ab_test, "_get_production_trades", side_effect=Exception("db error")):
            result = ab_test.get_results("error_test")
        assert result.status == "error"


# ============================================================
# PaperABTest stop & list tests
# ============================================================

class TestPaperABTestManagement:
    """Test A/B test stop and list."""

    def test_stop_test(self, ab_test):
        ab_test.setup_test("stop_me", {"ORB_ENABLED": False}, 7)
        stopped = ab_test.stop_test("stop_me")
        assert stopped is True
        data = json.loads((TEST_AB_DIR / "stop_me.json").read_text())
        assert data["status"] == "completed"
        assert data["end_date"] != ""

    def test_stop_nonexistent(self, ab_test):
        assert ab_test.stop_test("does_not_exist") is False

    def test_list_tests(self, ab_test):
        ab_test.setup_test("test_a", {"ORB_ENABLED": False}, 5)
        ab_test.setup_test("test_b", {"VWAP_BAND_STD": 3.0}, 10)
        tests = ab_test.list_tests()
        assert len(tests) == 2
        names = {t["name"] for t in tests}
        assert "test_a" in names
        assert "test_b" in names

    def test_list_tests_empty(self, ab_test):
        tests = ab_test.list_tests()
        assert tests == []


# ============================================================
# Shadow signal/trade recording tests
# ============================================================

class TestShadowRecording:
    """Test shadow signal and trade recording."""

    def test_record_shadow_signal(self, ab_test):
        ab_test.setup_test("sig_test", {}, 7)
        success = ab_test.record_shadow_signal("sig_test", {
            "symbol": "AAPL", "strategy": "ORB", "side": "buy"
        })
        assert success is True
        data = json.loads((TEST_AB_DIR / "sig_test.json").read_text())
        assert len(data["shadow_signals"]) == 1
        assert data["shadow_signals"][0]["symbol"] == "AAPL"

    def test_record_shadow_signal_inactive(self, ab_test):
        ab_test.setup_test("inactive_sig", {}, 7)
        ab_test.stop_test("inactive_sig")
        success = ab_test.record_shadow_signal("inactive_sig", {"symbol": "AAPL"})
        assert success is False

    def test_record_shadow_trade(self, ab_test):
        ab_test.setup_test("trade_test", {}, 7)
        success = ab_test.record_shadow_trade("trade_test", {
            "symbol": "MSFT", "pnl": 25.0
        })
        assert success is True
        data = json.loads((TEST_AB_DIR / "trade_test.json").read_text())
        assert len(data["shadow_trades"]) == 1
        assert data["shadow_trades"][0]["pnl"] == 25.0

    def test_record_shadow_trade_nonexistent(self, ab_test):
        success = ab_test.record_shadow_trade("nope", {"pnl": 10.0})
        assert success is False


# ============================================================
# Statistics computation tests
# ============================================================

class TestComputeStats:
    """Test _compute_stats static method."""

    def test_empty_trades(self):
        from ab_testing import PaperABTest
        stats = PaperABTest._compute_stats([])
        assert stats["signal_count"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["sharpe"] == 0.0
        assert stats["sortino"] == 0.0
        assert stats["max_drawdown"] == 0.0
        assert stats["total_pnl"] == 0.0

    def test_all_winners(self):
        from ab_testing import PaperABTest
        trades = [{"pnl": 10.0}, {"pnl": 20.0}, {"pnl": 5.0}]
        stats = PaperABTest._compute_stats(trades)
        assert stats["signal_count"] == 3
        assert stats["win_rate"] == 1.0
        assert stats["total_pnl"] == 35.0
        assert stats["sharpe"] > 0

    def test_all_losers(self):
        from ab_testing import PaperABTest
        trades = [{"pnl": -10.0}, {"pnl": -20.0}, {"pnl": -5.0}]
        stats = PaperABTest._compute_stats(trades)
        assert stats["win_rate"] == 0.0
        assert stats["total_pnl"] == -35.0
        assert stats["sharpe"] < 0

    def test_mixed_trades(self):
        from ab_testing import PaperABTest
        trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 20.0}, {"pnl": -3.0}]
        stats = PaperABTest._compute_stats(trades)
        assert stats["signal_count"] == 4
        assert stats["win_rate"] == 0.5
        assert stats["total_pnl"] == pytest.approx(22.0)

    def test_max_drawdown(self):
        from ab_testing import PaperABTest
        # sequence: +10, -20, +5 => cumulative: 10, -10, -5
        # peak=10, max dd from peak = 10-(-10) = 20
        trades = [{"pnl": 10.0}, {"pnl": -20.0}, {"pnl": 5.0}]
        stats = PaperABTest._compute_stats(trades)
        assert stats["max_drawdown"] == pytest.approx(20.0)

    def test_none_pnl_treated_as_zero(self):
        from ab_testing import PaperABTest
        trades = [{"pnl": None}, {"pnl": 10.0}]
        stats = PaperABTest._compute_stats(trades)
        assert stats["total_pnl"] == 10.0


# ============================================================
# Bootstrap significance tests
# ============================================================

class TestBootstrapSignificance:
    """Test bootstrap confidence interval calculation."""

    def test_empty_trades(self):
        from ab_testing import PaperABTest
        assert PaperABTest._bootstrap_significance([], []) is False
        assert PaperABTest._bootstrap_significance([{"pnl": 1}], []) is False
        assert PaperABTest._bootstrap_significance([], [{"pnl": 1}]) is False

    def test_clearly_significant(self):
        """Large difference should be significant."""
        from ab_testing import PaperABTest
        import random
        random.seed(42)
        trades_a = [{"pnl": 100.0 + i} for i in range(50)]
        trades_b = [{"pnl": -100.0 - i} for i in range(50)]
        result = PaperABTest._bootstrap_significance(trades_a, trades_b)
        assert result is True

    def test_not_significant_identical(self):
        """Identical distributions should not be significant."""
        from ab_testing import PaperABTest
        import random
        random.seed(42)
        trades = [{"pnl": float(i % 10 - 5)} for i in range(50)]
        result = PaperABTest._bootstrap_significance(trades, trades)
        assert result is False

    def test_single_trade_each(self):
        """Single trade comparison should work without error."""
        from ab_testing import PaperABTest
        import random
        random.seed(42)
        result = PaperABTest._bootstrap_significance(
            [{"pnl": 100.0}], [{"pnl": -100.0}]
        )
        # With single samples, bootstrap may or may not find significance
        assert isinstance(result, bool)


# ============================================================
# Config override application tests
# ============================================================

class TestConfigOverrides:
    """Test that config overrides are correctly applied during A/B tests."""

    def test_setup_stores_overrides(self, ab_test):
        ab_test.setup_test("override_test", {
            "ORB_ENABLED": False,
            "VWAP_BAND_STD": 3.0,
            "MAX_POSITIONS": 5,
        }, 14)
        data = json.loads((TEST_AB_DIR / "override_test.json").read_text())
        overrides = data["config_b_overrides"]
        assert overrides["ORB_ENABLED"] is False
        assert overrides["VWAP_BAND_STD"] == 3.0
        assert overrides["MAX_POSITIONS"] == 5
