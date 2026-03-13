"""Tests for OU process tools."""

import numpy as np
import pandas as pd
import pytest


class TestFitOUParams:
    def test_mean_reverting_series(self):
        """OU fit on synthetic mean-reverting series returns positive kappa."""
        from analytics.ou_tools import fit_ou_params

        np.random.seed(42)
        n = 200
        mu, kappa, sigma = 100.0, 0.1, 1.0
        prices = [mu]
        for _ in range(n - 1):
            dp = kappa * (mu - prices[-1]) + sigma * np.random.randn()
            prices.append(prices[-1] + dp)

        result = fit_ou_params(pd.Series(prices))
        assert result != {}
        assert result['kappa'] > 0
        assert result['half_life'] > 0
        assert abs(result['mu'] - mu) < 20  # Reasonable estimate

    def test_trending_series_fails(self):
        """OU fit on trending series returns empty (b >= 0)."""
        from analytics.ou_tools import fit_ou_params

        # Strong exponential uptrend — clearly non-mean-reverting
        prices = pd.Series(np.exp(np.linspace(0, 3, 200)))
        result = fit_ou_params(prices)
        assert result == {}

    def test_insufficient_data(self):
        """Returns empty dict for too-short series."""
        from analytics.ou_tools import fit_ou_params
        result = fit_ou_params(pd.Series([1, 2, 3]))
        assert result == {}


class TestComputeZscore:
    def test_at_mean(self):
        from analytics.ou_tools import compute_zscore
        assert compute_zscore(100.0, 100.0, 2.0) == 0.0

    def test_above_mean(self):
        from analytics.ou_tools import compute_zscore
        z = compute_zscore(104.0, 100.0, 2.0)
        assert z == 2.0

    def test_below_mean(self):
        from analytics.ou_tools import compute_zscore
        z = compute_zscore(97.0, 100.0, 2.0)
        assert z == -1.5

    def test_zero_sigma(self):
        from analytics.ou_tools import compute_zscore
        assert compute_zscore(105.0, 100.0, 0.0) == 0.0
