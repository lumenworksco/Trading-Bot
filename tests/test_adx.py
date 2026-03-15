"""Tests for V8 ADX trend strength indicator."""

import pytest
import numpy as np
import pandas as pd


class TestComputeADX:

    def test_trending_market_high_adx(self):
        """Strong uptrend should produce high ADX (>25)."""
        from analytics.adx import compute_adx

        n = 100
        # Simulate strong uptrend
        np.random.seed(42)
        base = np.cumsum(np.random.uniform(0.5, 1.5, n)) + 100
        highs = pd.Series(base + np.random.uniform(0, 2, n))
        lows = pd.Series(base - np.random.uniform(0, 1, n))
        closes = pd.Series(base + np.random.uniform(-0.5, 1, n))

        adx = compute_adx(highs, lows, closes, period=14)
        assert adx is not None
        assert adx > 20  # Trending should have elevated ADX

    def test_ranging_market_low_adx(self):
        """Ranging/choppy market should produce low ADX."""
        from analytics.adx import compute_adx

        n = 100
        np.random.seed(42)
        # Mean-reverting around 100
        base = 100 + np.sin(np.linspace(0, 20, n)) * 2 + np.random.normal(0, 0.3, n)
        highs = pd.Series(base + np.random.uniform(0, 1, n))
        lows = pd.Series(base - np.random.uniform(0, 1, n))
        closes = pd.Series(base + np.random.uniform(-0.3, 0.3, n))

        adx = compute_adx(highs, lows, closes, period=14)
        assert adx is not None
        assert adx < 30  # Ranging should have low ADX

    def test_insufficient_data_returns_none(self):
        """Too few bars should return None."""
        from analytics.adx import compute_adx

        n = 10  # Less than 2*period+1 = 29
        highs = pd.Series(np.random.uniform(100, 102, n))
        lows = pd.Series(np.random.uniform(98, 100, n))
        closes = pd.Series(np.random.uniform(99, 101, n))

        adx = compute_adx(highs, lows, closes, period=14)
        assert adx is None

    def test_adx_range_0_to_100(self):
        """ADX should always be between 0 and 100."""
        from analytics.adx import compute_adx

        n = 100
        np.random.seed(123)
        base = np.cumsum(np.random.normal(0, 1, n)) + 100
        highs = pd.Series(base + abs(np.random.normal(0, 1, n)))
        lows = pd.Series(base - abs(np.random.normal(0, 1, n)))
        closes = pd.Series(base)

        adx = compute_adx(highs, lows, closes, period=14)
        assert adx is not None
        assert 0 <= adx <= 100

    def test_custom_period(self):
        """ADX should work with custom period."""
        from analytics.adx import compute_adx

        n = 100
        np.random.seed(42)
        base = np.cumsum(np.random.uniform(0.2, 0.8, n)) + 100
        highs = pd.Series(base + 1)
        lows = pd.Series(base - 1)
        closes = pd.Series(base)

        adx_7 = compute_adx(highs, lows, closes, period=7)
        adx_14 = compute_adx(highs, lows, closes, period=14)

        assert adx_7 is not None
        assert adx_14 is not None

    def test_flat_market(self):
        """Completely flat market should have low ADX."""
        from analytics.adx import compute_adx

        n = 100
        highs = pd.Series([101.0] * n)
        lows = pd.Series([99.0] * n)
        closes = pd.Series([100.0] * n)

        adx = compute_adx(highs, lows, closes, period=14)
        # Flat market: +DM and -DM are 0, so DX is 0, ADX approaches 0
        assert adx is not None
        assert adx < 5
