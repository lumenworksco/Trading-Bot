"""Tests for consistency score computation."""

import pytest


class TestConsistencyScore:
    def test_perfect_returns(self):
        """All positive days + high Sharpe + no drawdown = near 100."""
        from analytics.consistency_score import compute_consistency_score
        daily = [0.01] * 30  # 1% every day
        score = compute_consistency_score(daily, sharpe=3.0, max_drawdown=0.0)
        assert score >= 90

    def test_terrible_returns(self):
        """All negative days + negative Sharpe + deep drawdown = near 0."""
        from analytics.consistency_score import compute_consistency_score
        daily = [-0.02] * 30
        score = compute_consistency_score(daily, sharpe=-1.0, max_drawdown=-0.10)
        assert score <= 15

    def test_mediocre_returns(self):
        """Mixed days should give middle score."""
        from analytics.consistency_score import compute_consistency_score
        daily = [0.005, -0.003] * 15  # 50% win rate
        score = compute_consistency_score(daily, sharpe=0.5, max_drawdown=-0.02)
        assert 30 <= score <= 70

    def test_insufficient_data(self):
        """Less than 5 days returns 50 (neutral)."""
        from analytics.consistency_score import compute_consistency_score
        assert compute_consistency_score([0.01, 0.02]) == 50.0

    def test_score_clamped(self):
        """Score is always between 0 and 100."""
        from analytics.consistency_score import compute_consistency_score
        score = compute_consistency_score([0.05] * 30, sharpe=5.0, max_drawdown=0.0)
        assert 0 <= score <= 100
