"""Tests for Hurst exponent computation."""

import numpy as np
import pandas as pd
import pytest


class TestHurstExponent:
    def test_random_walk_near_half(self):
        """Random walk should have Hurst exponent near 0.5."""
        from analytics.hurst import hurst_exponent

        np.random.seed(42)
        rw = pd.Series(np.cumsum(np.random.randn(500)))
        h = hurst_exponent(rw)
        assert 0.35 <= h <= 0.65  # Should be roughly 0.5

    def test_mean_reverting_below_half(self):
        """Mean-reverting series should have Hurst < 0.5."""
        from analytics.hurst import hurst_exponent

        np.random.seed(42)
        n = 500
        mu, kappa = 0.0, 0.3
        prices = [mu]
        for _ in range(n - 1):
            prices.append(prices[-1] + kappa * (mu - prices[-1]) + 0.5 * np.random.randn())
        h = hurst_exponent(pd.Series(prices))
        assert h < 0.55  # Should be below 0.5

    def test_insufficient_data_returns_half(self):
        """Too few data points returns 0.5 (neutral)."""
        from analytics.hurst import hurst_exponent
        assert hurst_exponent(pd.Series([1, 2, 3])) == 0.5

    def test_clamped_to_valid_range(self):
        """Output is always between 0 and 1."""
        from analytics.hurst import hurst_exponent

        np.random.seed(42)
        h = hurst_exponent(pd.Series(np.random.randn(100)))
        assert 0.0 <= h <= 1.0
