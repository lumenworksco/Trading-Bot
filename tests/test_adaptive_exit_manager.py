"""Tests for adaptive_exit_manager.AdaptiveExitManager."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_exit_manager import AdaptiveExitManager

ET = ZoneInfo("America/New_York")


# ------------------------------------------------------------------ #
#  get_exit_params
# ------------------------------------------------------------------ #

class TestGetExitParams:

    def test_calm_market_params(self):
        """VIX < 15 should return calm-market parameters."""
        params = AdaptiveExitManager.get_exit_params(vix=12.0,
                                                      half_life_hours=6.0)
        assert params['z_exit'] == 0.3
        assert params['time_mult'] == 3.0
        assert params['partial_z'] == 0.8

    def test_crisis_market_params(self):
        """VIX > 35 should return crisis parameters."""
        params = AdaptiveExitManager.get_exit_params(vix=40.0,
                                                      half_life_hours=6.0)
        assert params['z_exit'] == 0.8
        assert params['time_mult'] == 0.5
        assert params['partial_z'] == 1.5


# ------------------------------------------------------------------ #
#  should_exit
# ------------------------------------------------------------------ #

def _ou_params(mu=100.0, sigma=2.0, half_life=6.0):
    return {'mu': mu, 'sigma': sigma, 'half_life': half_life}


class TestShouldExit:

    def test_full_reversion_exit(self):
        """Price near the mean should trigger full reversion exit."""
        mgr = AdaptiveExitManager()
        entry = datetime.now(ET) - timedelta(hours=1)
        # current_price very close to mu -> z ~ 0
        reason, exit_type = mgr.should_exit(
            position_side='buy',
            position_entry_time=entry,
            current_price=100.1,
            ou_params=_ou_params(mu=100.0, sigma=2.0),
            vix=20.0,
            partial_exits=0,
        )
        assert exit_type == 'full'
        assert reason == 'full_reversion'

    def test_partial_exit(self):
        """Price within partial_z but beyond z_exit, no prior partials."""
        mgr = AdaptiveExitManager()
        entry = datetime.now(ET) - timedelta(hours=1)
        # VIX 20 -> z_exit=0.2, partial_z=0.5
        # z = (100.8 - 100) / 2 = 0.4 -> within partial_z but > z_exit
        reason, exit_type = mgr.should_exit(
            position_side='buy',
            position_entry_time=entry,
            current_price=100.8,
            ou_params=_ou_params(mu=100.0, sigma=2.0),
            vix=20.0,
            partial_exits=0,
        )
        assert exit_type == 'partial'
        assert reason == 'partial_reversion'

    def test_time_stop_exit(self):
        """Position held longer than half_life * time_mult triggers exit."""
        mgr = AdaptiveExitManager()
        # VIX 20 -> time_mult=2.0, half_life=6h -> max 12h
        entry = datetime.now(ET) - timedelta(hours=13)
        # z far enough from zero to skip reversion checks
        reason, exit_type = mgr.should_exit(
            position_side='buy',
            position_entry_time=entry,
            current_price=103.0,
            ou_params=_ou_params(mu=100.0, sigma=2.0, half_life=6.0),
            vix=20.0,
            partial_exits=1,
        )
        assert exit_type == 'full'
        assert reason == 'time_stop'

    def test_hold_when_no_conditions_met(self):
        """No exit condition met should return hold."""
        mgr = AdaptiveExitManager()
        entry = datetime.now(ET) - timedelta(hours=1)
        # z = (101.5 - 100) / 2 = 0.75 — beyond partial_z for normal VIX
        # VIX 20: partial_z=0.5, z_exit=0.2 — z > partial_z
        # time_mult=2.0, half_life=6 -> max=12h, only 1h in
        # already did partial exit -> partial won't trigger
        reason, exit_type = mgr.should_exit(
            position_side='buy',
            position_entry_time=entry,
            current_price=101.5,
            ou_params=_ou_params(mu=100.0, sigma=2.0, half_life=6.0),
            vix=20.0,
            partial_exits=1,
        )
        assert exit_type == 'hold'
        assert reason == 'hold'
