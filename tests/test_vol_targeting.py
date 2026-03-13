"""Tests for volatility targeting risk engine."""

import pytest
from unittest.mock import patch


class TestVolScalar:
    def test_normal_conditions(self):
        """Normal VIX and vol should give scalar near 1.0."""
        with patch.dict('config.__dict__', {
            'VOL_TARGET_DAILY': 0.01,
            'VOL_TARGET_MAX': 0.015,
            'VOL_SCALAR_MIN': 0.3,
            'VOL_SCALAR_MAX': 1.5,
        }):
            from risk.vol_targeting import VolatilityTargetingRiskEngine
            engine = VolatilityTargetingRiskEngine()
            scalar = engine.compute_vol_scalar(vix=20.0, portfolio_atr_vol=0.01, rolling_pnl_std=0.01)
            assert 0.5 <= scalar <= 1.5

    def test_high_vol_scales_down(self):
        """High volatility should scale position sizes down."""
        with patch.dict('config.__dict__', {
            'VOL_TARGET_DAILY': 0.01,
            'VOL_TARGET_MAX': 0.015,
            'VOL_SCALAR_MIN': 0.3,
            'VOL_SCALAR_MAX': 1.5,
        }):
            from risk.vol_targeting import VolatilityTargetingRiskEngine
            engine = VolatilityTargetingRiskEngine()
            scalar = engine.compute_vol_scalar(vix=40.0, portfolio_atr_vol=0.03, rolling_pnl_std=0.025)
            assert scalar < 0.8

    def test_low_vol_scales_up(self):
        """Low volatility should scale position sizes up (capped at 1.5)."""
        with patch.dict('config.__dict__', {
            'VOL_TARGET_DAILY': 0.01,
            'VOL_TARGET_MAX': 0.015,
            'VOL_SCALAR_MIN': 0.3,
            'VOL_SCALAR_MAX': 1.5,
        }):
            from risk.vol_targeting import VolatilityTargetingRiskEngine
            engine = VolatilityTargetingRiskEngine()
            scalar = engine.compute_vol_scalar(vix=10.0, portfolio_atr_vol=0.003, rolling_pnl_std=0.002)
            assert scalar >= 1.0
            assert scalar <= 1.5


class TestPositionSizing:
    def test_basic_sizing(self):
        """Position size based on risk per trade."""
        with patch.dict('config.__dict__', {
            'RISK_PER_TRADE_PCT': 0.008,
            'STRATEGY_ALLOCATIONS': {'STAT_MR': 0.60},
            'SHORT_SIZE_MULTIPLIER': 0.75,
            'MAX_POSITION_PCT': 0.07,
            'MIN_POSITION_VALUE': 100,
        }):
            from risk.vol_targeting import VolatilityTargetingRiskEngine
            engine = VolatilityTargetingRiskEngine()
            qty = engine.calculate_position_size(
                equity=100000, entry_price=150.0, stop_price=148.0,
                vol_scalar=1.0, strategy='STAT_MR', pnl_lock_mult=1.0
            )
            assert qty > 0
            assert qty * 150 <= 100000 * 0.07  # Doesn't exceed max position

    def test_zero_equity(self):
        from risk.vol_targeting import VolatilityTargetingRiskEngine
        engine = VolatilityTargetingRiskEngine()
        assert engine.calculate_position_size(0, 100.0, 99.0) == 0

    def test_zero_risk(self):
        from risk.vol_targeting import VolatilityTargetingRiskEngine
        engine = VolatilityTargetingRiskEngine()
        assert engine.calculate_position_size(100000, 100.0, 100.0) == 0

    def test_pnl_lock_reduces_size(self):
        """Gain lock reduces position size to 30%."""
        with patch.dict('config.__dict__', {
            'RISK_PER_TRADE_PCT': 0.008,
            'STRATEGY_ALLOCATIONS': {'STAT_MR': 0.60},
            'SHORT_SIZE_MULTIPLIER': 0.75,
            'MAX_POSITION_PCT': 0.50,  # High cap so it doesn't mask the pnl_lock effect
            'MIN_POSITION_VALUE': 100,
        }):
            from risk.vol_targeting import VolatilityTargetingRiskEngine
            engine = VolatilityTargetingRiskEngine()
            full = engine.calculate_position_size(
                100000, 150.0, 148.0, vol_scalar=1.0, strategy='STAT_MR', pnl_lock_mult=1.0
            )
            locked = engine.calculate_position_size(
                100000, 150.0, 148.0, vol_scalar=1.0, strategy='STAT_MR', pnl_lock_mult=0.3
            )
            assert locked < full
            assert locked > 0
