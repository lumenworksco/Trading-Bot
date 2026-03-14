"""Tests for walk_forward.WalkForwardValidator."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from walk_forward import WalkForwardValidator


# ------------------------------------------------------------------ #
#  compute_sharpe
# ------------------------------------------------------------------ #

class TestComputeSharpe:

    def test_compute_sharpe_positive(self):
        """Sharpe with a clear positive-mean return series."""
        returns = [0.01, 0.02, 0.015, 0.005, 0.012, 0.018, 0.009]
        sharpe = WalkForwardValidator.compute_sharpe(returns)
        assert sharpe > 0, "Sharpe should be positive for all-positive returns"

    def test_compute_sharpe_insufficient_data(self):
        """Fewer than 5 data points should return 0.0."""
        assert WalkForwardValidator.compute_sharpe([0.01, 0.02]) == 0.0
        assert WalkForwardValidator.compute_sharpe([]) == 0.0


# ------------------------------------------------------------------ #
#  validate_strategy
# ------------------------------------------------------------------ #

def _make_trades(pnl_pcts: list[float], strategy: str = "STAT_MR") -> list[dict]:
    """Build a list of trade dicts with the given pnl_pcts."""
    trades = []
    for i, pnl in enumerate(pnl_pcts):
        trades.append({
            'pnl_pct': pnl,
            'strategy': strategy,
            'entry_time': f'2026-03-{1 + i:02d}T10:00:00',
            'exit_time': f'2026-03-{1 + i:02d}T15:00:00',
        })
    return trades


class TestValidateStrategy:

    def test_validate_strategy_demote(self):
        """OOS Sharpe below WALK_FORWARD_MIN_SHARPE should demote."""
        # Alternate positive/negative so Sharpe is near zero
        pnl_pcts = [0.01, -0.01, 0.005, -0.005, 0.002, -0.002,
                     0.001, -0.001, 0.003, -0.003, 0.002, -0.002]
        trades = _make_trades(pnl_pcts)

        validator = WalkForwardValidator()
        with patch('config.WALK_FORWARD_MIN_SHARPE', 0.3):
            result = validator.validate_strategy('STAT_MR', trades)

        assert result['recommendation'] == 'demote'
        assert result['total_trades'] > 0

    def test_validate_strategy_maintain(self):
        """Moderate OOS Sharpe should maintain."""
        # Slightly positive bias in OOS half
        pnl_pcts = [
            # in-sample (first half)
            0.005, -0.003, 0.004, -0.002, 0.006,
            # out-of-sample (second half) — moderate edge
            0.008, 0.003, -0.001, 0.006, 0.004,
        ]
        trades = _make_trades(pnl_pcts)

        validator = WalkForwardValidator()
        with patch('config.WALK_FORWARD_MIN_SHARPE', 0.3):
            result = validator.validate_strategy('STAT_MR', trades)

        assert result['recommendation'] in ('maintain', 'promote')
        assert result['sharpe'] >= 0.3
