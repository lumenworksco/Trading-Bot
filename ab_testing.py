"""Live Paper A/B Testing — run two configurations simultaneously for comparison."""

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import config

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

AB_TESTS_DIR = Path("tests_ab")


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""
    name: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_days: int = 0
    status: str = "pending"
    config_a_stats: dict = field(default_factory=lambda: {
        "signal_count": 0, "win_rate": 0.0, "sharpe": 0.0,
        "sortino": 0.0, "max_drawdown": 0.0, "total_pnl": 0.0,
    })
    config_b_stats: dict = field(default_factory=lambda: {
        "signal_count": 0, "win_rate": 0.0, "sharpe": 0.0,
        "sortino": 0.0, "max_drawdown": 0.0, "total_pnl": 0.0,
    })
    delta: dict = field(default_factory=lambda: {
        "signal_count": 0, "win_rate": 0.0, "sharpe": 0.0,
        "sortino": 0.0, "max_drawdown": 0.0, "total_pnl": 0.0,
    })
    significant: bool = False


# ============================================================
# PaperABTest
# ============================================================

class PaperABTest:
    """
    Runs two configurations simultaneously on paper.
    Config A = current production. Config B = proposed change.
    Both see the same market data and generate signals independently.
    Only Config A actually executes (Config B is shadow-only).
    """

    def __init__(self):
        AB_TESTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def setup_test(self, name: str, config_b_overrides: dict,
                   duration_days: int = 7) -> bool:
        """Start a new A/B test. Config B signals go to shadow_trades table."""
        if not getattr(config, "AB_TESTING_ENABLED", True):
            logger.info("A/B testing disabled via config")
            return False

        try:
            test_file = AB_TESTS_DIR / f"{name}.json"
            now = datetime.now(ET)

            test_config = {
                "name": name,
                "start_date": now.strftime("%Y-%m-%d"),
                "end_date": "",
                "duration_days": duration_days,
                "status": "active",
                "config_b_overrides": config_b_overrides,
                "shadow_signals": [],
                "shadow_trades": [],
                "created_at": now.isoformat(),
            }

            test_file.write_text(json.dumps(test_config, indent=2))
            logger.info(f"A/B test '{name}' started with duration {duration_days} days")
            return True

        except Exception as e:
            logger.error(f"Failed to setup A/B test '{name}': {e}")
            return False

    def get_results(self, name: str) -> ABTestResult:
        """
        Compare A vs B: signal count, win rate, Sharpe, Sortino, max drawdown.
        Statistical significance via bootstrap confidence interval.
        """
        result = ABTestResult(name=name)

        try:
            test_data = self._load_test(name)
            if not test_data:
                result.status = "not_found"
                return result

            result.start_date = test_data.get("start_date", "")
            result.end_date = test_data.get("end_date", "")
            result.duration_days = test_data.get("duration_days", 0)
            result.status = test_data.get("status", "unknown")

            # Compute Config A stats from production trades
            config_a_trades = self._get_production_trades(result.start_date)
            result.config_a_stats = self._compute_stats(config_a_trades)

            # Compute Config B stats from shadow trades
            shadow_trades = test_data.get("shadow_trades", [])
            result.config_b_stats = self._compute_stats(shadow_trades)

            # Compute deltas (A - B)
            result.delta = {
                key: result.config_a_stats.get(key, 0) - result.config_b_stats.get(key, 0)
                for key in result.config_a_stats
            }

            # Bootstrap significance test
            result.significant = self._bootstrap_significance(
                config_a_trades, shadow_trades
            )

        except Exception as e:
            logger.error(f"Failed to get results for test '{name}': {e}")
            result.status = "error"

        return result

    def stop_test(self, name: str) -> bool:
        """Stop an active A/B test."""
        try:
            test_data = self._load_test(name)
            if not test_data:
                return False

            test_data["status"] = "completed"
            test_data["end_date"] = datetime.now(ET).strftime("%Y-%m-%d")
            self._save_test(name, test_data)
            logger.info(f"A/B test '{name}' stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop test '{name}': {e}")
            return False

    def list_tests(self) -> list[dict]:
        """List all A/B tests with basic info."""
        tests = []
        try:
            for test_file in AB_TESTS_DIR.glob("*.json"):
                try:
                    data = json.loads(test_file.read_text())
                    tests.append({
                        "name": data.get("name", test_file.stem),
                        "status": data.get("status", "unknown"),
                        "start_date": data.get("start_date", ""),
                        "duration_days": data.get("duration_days", 0),
                    })
                except (json.JSONDecodeError, IOError):
                    continue
        except Exception as e:
            logger.error(f"Failed to list tests: {e}")
        return tests

    def record_shadow_signal(self, name: str, signal: dict) -> bool:
        """Record a shadow signal for Config B during a live test."""
        try:
            test_data = self._load_test(name)
            if not test_data or test_data.get("status") != "active":
                return False

            signal["timestamp"] = datetime.now(ET).isoformat()
            test_data.setdefault("shadow_signals", []).append(signal)
            self._save_test(name, test_data)
            return True

        except Exception as e:
            logger.error(f"Failed to record shadow signal for '{name}': {e}")
            return False

    def record_shadow_trade(self, name: str, trade: dict) -> bool:
        """Record a shadow trade result for Config B."""
        try:
            test_data = self._load_test(name)
            if not test_data or test_data.get("status") != "active":
                return False

            test_data.setdefault("shadow_trades", []).append(trade)
            self._save_test(name, test_data)
            return True

        except Exception as e:
            logger.error(f"Failed to record shadow trade for '{name}': {e}")
            return False

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _load_test(self, name: str) -> dict | None:
        """Load test config from JSON file."""
        test_file = AB_TESTS_DIR / f"{name}.json"
        if not test_file.exists():
            return None
        try:
            return json.loads(test_file.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load test '{name}': {e}")
            return None

    def _save_test(self, name: str, data: dict):
        """Save test config to JSON file."""
        test_file = AB_TESTS_DIR / f"{name}.json"
        test_file.write_text(json.dumps(data, indent=2))

    def _get_production_trades(self, start_date: str) -> list[dict]:
        """Get production trades since start_date from database."""
        try:
            import database
            all_trades = database.get_all_trades()
            return [
                t for t in all_trades
                if str(t.get("exit_time", "")) >= start_date
            ]
        except Exception as e:
            logger.warning(f"Failed to load production trades: {e}")
            return []

    @staticmethod
    def _compute_stats(trades: list[dict]) -> dict:
        """Compute performance statistics from a list of trade dicts."""
        stats = {
            "signal_count": len(trades),
            "win_rate": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
        }

        if not trades:
            return stats

        pnls = [t.get("pnl", 0.0) or 0.0 for t in trades]
        stats["total_pnl"] = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        stats["win_rate"] = wins / len(pnls) if pnls else 0.0

        # Sharpe
        if len(pnls) >= 2:
            mean_pnl = sum(pnls) / len(pnls)
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std = math.sqrt(variance) if variance > 0 else 0.0
            stats["sharpe"] = mean_pnl / std if std > 0 else 0.0

        # Sortino (downside deviation only)
        if len(pnls) >= 2:
            mean_pnl = sum(pnls) / len(pnls)
            downside = [min(p - mean_pnl, 0) ** 2 for p in pnls]
            down_dev = math.sqrt(sum(downside) / (len(downside) - 1)) if len(downside) > 1 else 0.0
            stats["sortino"] = mean_pnl / down_dev if down_dev > 0 else 0.0

        # Max drawdown
        if pnls:
            cumulative = 0.0
            peak = 0.0
            max_dd = 0.0
            for p in pnls:
                cumulative += p
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd
            stats["max_drawdown"] = max_dd

        return stats

    @staticmethod
    def _bootstrap_significance(trades_a: list[dict], trades_b: list[dict],
                                n_iterations: int = 1000,
                                confidence: float = 0.95) -> bool:
        """
        Test if the difference in mean P&L between A and B is significant
        using bootstrap confidence intervals.
        """
        pnls_a = [t.get("pnl", 0.0) or 0.0 for t in trades_a]
        pnls_b = [t.get("pnl", 0.0) or 0.0 for t in trades_b]

        if not pnls_a or not pnls_b:
            return False

        observed_diff = (sum(pnls_a) / len(pnls_a)) - (sum(pnls_b) / len(pnls_b))

        # Bootstrap the difference in means
        diffs = []
        for _ in range(n_iterations):
            sample_a = random.choices(pnls_a, k=len(pnls_a))
            sample_b = random.choices(pnls_b, k=len(pnls_b))
            mean_a = sum(sample_a) / len(sample_a)
            mean_b = sum(sample_b) / len(sample_b)
            diffs.append(mean_a - mean_b)

        diffs.sort()
        alpha = 1 - confidence
        lower_idx = int(alpha / 2 * n_iterations)
        upper_idx = int((1 - alpha / 2) * n_iterations)

        if lower_idx >= len(diffs) or upper_idx >= len(diffs):
            return False

        lower_bound = diffs[lower_idx]
        upper_bound = diffs[upper_idx]

        # Significant if CI doesn't contain zero
        return lower_bound > 0 or upper_bound < 0
