"""Tests for analytics.intraday_seasonality — IntradaySeasonality class."""

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from analytics.intraday_seasonality import IntradaySeasonality

ET = ZoneInfo("America/New_York")


@pytest.fixture
def seasonality():
    return IntradaySeasonality()


# ===================================================================
# Helper to build ET-aware datetimes at a specific time
# ===================================================================

def _dt(hour, minute=0, second=0):
    """Return a timezone-aware datetime on a regular trading day."""
    return datetime(2026, 3, 13, hour, minute, second, tzinfo=ET)


# ===================================================================
# get_current_window — one test per window
# ===================================================================

class TestGetCurrentWindow:
    def test_open_auction(self, seasonality):
        assert seasonality.get_current_window(_dt(9, 30)) == "open_auction"
        assert seasonality.get_current_window(_dt(9, 44, 59)) == "open_auction"

    def test_opening_drive(self, seasonality):
        assert seasonality.get_current_window(_dt(9, 45)) == "opening_drive"
        assert seasonality.get_current_window(_dt(10, 15)) == "opening_drive"

    def test_mid_morning(self, seasonality):
        assert seasonality.get_current_window(_dt(10, 30)) == "mid_morning"
        assert seasonality.get_current_window(_dt(11, 0)) == "mid_morning"

    def test_lunch_lull(self, seasonality):
        assert seasonality.get_current_window(_dt(11, 30)) == "lunch_lull"
        assert seasonality.get_current_window(_dt(13, 0)) == "lunch_lull"

    def test_afternoon(self, seasonality):
        assert seasonality.get_current_window(_dt(14, 0)) == "afternoon"
        assert seasonality.get_current_window(_dt(15, 15)) == "afternoon"

    def test_close_auction(self, seasonality):
        assert seasonality.get_current_window(_dt(15, 30)) == "close_auction"
        assert seasonality.get_current_window(_dt(15, 59)) == "close_auction"

    def test_before_market_open(self, seasonality):
        assert seasonality.get_current_window(_dt(9, 0)) is None

    def test_after_market_close(self, seasonality):
        assert seasonality.get_current_window(_dt(16, 0)) is None
        assert seasonality.get_current_window(_dt(17, 0)) is None

    def test_midnight(self, seasonality):
        assert seasonality.get_current_window(_dt(0, 0)) is None


# ===================================================================
# get_window_score — strategy-specific scoring
# ===================================================================

class TestGetWindowScore:
    def test_orb_disabled_in_lunch_lull(self, seasonality):
        """ORB should return 0.0 during lunch lull."""
        score = seasonality.get_window_score(_dt(12, 0), "ORB")
        assert score == 0.0

    def test_orb_disabled_in_afternoon(self, seasonality):
        """ORB should return 0.0 in the afternoon."""
        score = seasonality.get_window_score(_dt(14, 30), "ORB")
        assert score == 0.0

    def test_orb_boosted_in_opening_drive(self, seasonality):
        """ORB should return 1.5 during opening drive."""
        score = seasonality.get_window_score(_dt(10, 0), "ORB")
        assert score == 1.5

    def test_stat_mr_boosted_in_lunch_lull(self, seasonality):
        """STAT_MR should return 1.3 during lunch lull."""
        score = seasonality.get_window_score(_dt(12, 0), "STAT_MR")
        assert score == 1.3

    def test_vwap_boosted_in_lunch_lull(self, seasonality):
        """VWAP should return 1.3 during lunch lull."""
        score = seasonality.get_window_score(_dt(13, 0), "VWAP")
        assert score == 1.3

    def test_pead_always_neutral(self, seasonality):
        """PEAD scores 1.0 in every window."""
        for hour in [10, 11, 12, 14, 15]:
            score = seasonality.get_window_score(_dt(hour, 30), "PEAD")
            assert score == 1.0, f"PEAD should be 1.0 at {hour}:30"

    def test_unknown_strategy_returns_1(self, seasonality):
        """Unknown strategies should default to 1.0."""
        score = seasonality.get_window_score(_dt(12, 0), "UNKNOWN_STRAT")
        assert score == 1.0

    def test_outside_market_hours_returns_1(self, seasonality):
        """Outside market hours should return 1.0."""
        score = seasonality.get_window_score(_dt(8, 0), "ORB")
        assert score == 1.0

    def test_micro_mom_opening_drive(self, seasonality):
        """MICRO_MOM should be boosted (1.3) in opening drive."""
        score = seasonality.get_window_score(_dt(10, 0), "MICRO_MOM")
        assert score == 1.3

    def test_kalman_pairs_lunch_lull(self, seasonality):
        """KALMAN_PAIRS gets a slight boost (1.1) in lunch lull."""
        score = seasonality.get_window_score(_dt(12, 30), "KALMAN_PAIRS")
        assert score == 1.1


# ===================================================================
# Open auction blocking
# ===================================================================

class TestOpenAuctionBlock:
    def test_open_auction_blocked_by_default(self, seasonality, override_config):
        """With SEASONALITY_OPEN_AUCTION_BLOCK=True (default), all strategies
        get 0.0 during open auction."""
        with override_config(SEASONALITY_OPEN_AUCTION_BLOCK=True):
            for strat in ["STAT_MR", "VWAP", "ORB", "MICRO_MOM", "PEAD"]:
                score = seasonality.get_window_score(_dt(9, 35), strat)
                assert score == 0.0, f"{strat} should be blocked in open auction"

    def test_open_auction_not_blocked_when_disabled(self, seasonality, override_config):
        """With SEASONALITY_OPEN_AUCTION_BLOCK=False, strategies use their
        natural scores during open auction."""
        with override_config(SEASONALITY_OPEN_AUCTION_BLOCK=False):
            score = seasonality.get_window_score(_dt(9, 35), "PEAD")
            assert score == 1.0  # PEAD is always 1.0

            score = seasonality.get_window_score(_dt(9, 35), "STAT_MR")
            assert score == 0.3  # STAT_MR natural open_auction score


# ===================================================================
# Feature toggle
# ===================================================================

class TestFeatureToggle:
    def test_disabled_returns_neutral(self, seasonality, override_config):
        """When INTRADAY_SEASONALITY_ENABLED=False, always return 1.0."""
        with override_config(INTRADAY_SEASONALITY_ENABLED=False):
            # Even ORB in lunch lull (normally 0.0) should return 1.0
            score = seasonality.get_window_score(_dt(12, 0), "ORB")
            assert score == 1.0


# ===================================================================
# Score clamping
# ===================================================================

class TestScoreClamping:
    def test_all_scores_in_valid_range(self, seasonality, override_config):
        """Every strategy/window combination must produce a score in [0.0, 1.5]."""
        with override_config(SEASONALITY_OPEN_AUCTION_BLOCK=False):
            for strat in IntradaySeasonality.STRATEGY_WINDOW_SCORES:
                for window_name, (start, _end) in IntradaySeasonality.WINDOWS.items():
                    dt = datetime(2026, 3, 13, start.hour, start.minute, tzinfo=ET)
                    score = seasonality.get_window_score(dt, strat)
                    assert 0.0 <= score <= 1.5, (
                        f"{strat}/{window_name} score {score} out of range"
                    )

    def test_adaptive_scores_clamped(self, seasonality):
        """Adaptive scores that exceed the range should be clamped."""
        seasonality._adaptive_scores = {
            "STAT_MR": {"mid_morning": 2.5}  # Over the max of 1.5
        }
        score = seasonality.get_window_score(_dt(11, 0), "STAT_MR")
        assert score == 1.5

        seasonality._adaptive_scores = {
            "STAT_MR": {"mid_morning": -0.5}  # Below the min of 0.0
        }
        score = seasonality.get_window_score(_dt(11, 0), "STAT_MR")
        assert score == 0.0


# ===================================================================
# update_from_data placeholder
# ===================================================================

class TestUpdateFromData:
    def test_noop_when_disabled(self, seasonality, override_config):
        """update_from_data should be a no-op when adaptive learning is off."""
        with override_config(SEASONALITY_ADAPTIVE_LEARNING=False):
            seasonality.update_from_data(None)
            assert seasonality._adaptive_scores is None
