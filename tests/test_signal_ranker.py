"""Tests for analytics.signal_ranker — signal ranking and expected value."""

import numpy as np
import pandas as pd
import pytest

from conftest import _make_signal
from analytics.hmm_regime import MarketRegimeState, STRATEGY_REGIME_AFFINITY
from analytics.signal_ranker import SignalRanker


@pytest.fixture
def ranker():
    return SignalRanker()


@pytest.fixture
def trade_history_df():
    """Fabricate a trade history DataFrame with known win/loss stats."""
    rows = []
    # 70 % win rate for AAPL/ORB (7 wins, 3 losses out of 10)
    for i in range(7):
        rows.append({"symbol": "AAPL", "strategy": "ORB", "pnl_pct": 0.02})
    for i in range(3):
        rows.append({"symbol": "AAPL", "strategy": "ORB", "pnl_pct": -0.01})
    # 40 % win rate for MSFT/VWAP (4 wins, 6 losses out of 10)
    for i in range(4):
        rows.append({"symbol": "MSFT", "strategy": "VWAP", "pnl_pct": 0.015})
    for i in range(6):
        rows.append({"symbol": "MSFT", "strategy": "VWAP", "pnl_pct": -0.012})
    return pd.DataFrame(rows)


@pytest.fixture
def bull_regime_probs():
    """Regime probabilities dominated by LOW_VOL_BULL."""
    return {
        MarketRegimeState.LOW_VOL_BULL: 0.8,
        MarketRegimeState.HIGH_VOL_BULL: 0.1,
        MarketRegimeState.LOW_VOL_BEAR: 0.05,
        MarketRegimeState.HIGH_VOL_BEAR: 0.0,
        MarketRegimeState.MEAN_REVERTING: 0.05,
    }


@pytest.fixture
def bear_regime_probs():
    """Regime probabilities dominated by HIGH_VOL_BEAR."""
    return {
        MarketRegimeState.LOW_VOL_BULL: 0.0,
        MarketRegimeState.HIGH_VOL_BULL: 0.05,
        MarketRegimeState.LOW_VOL_BEAR: 0.1,
        MarketRegimeState.HIGH_VOL_BEAR: 0.8,
        MarketRegimeState.MEAN_REVERTING: 0.05,
    }


# ===================================================================
# Basic rank behaviour
# ===================================================================

class TestRankBasic:

    def test_rank_returns_list(self, ranker):
        signals = [_make_signal(symbol="AAPL"), _make_signal(symbol="MSFT")]
        result = ranker.rank(signals)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_rank_returns_sorted_by_score(self, ranker):
        """Higher R:R should score higher (confidence component)."""
        low_rr = _make_signal(symbol="A", entry_price=100, take_profit=101, stop_loss=99)
        high_rr = _make_signal(symbol="B", entry_price=100, take_profit=110, stop_loss=99)
        result = ranker.rank([low_rr, high_rr])
        # high_rr should come first
        assert result[0].symbol == "B"
        assert result[1].symbol == "A"

    def test_rank_empty_list(self, ranker):
        assert ranker.rank([]) == []

    def test_rank_single_signal(self, ranker):
        sig = _make_signal()
        result = ranker.rank([sig])
        assert len(result) == 1
        assert "[score=" in result[0].reason

    def test_score_appended_to_reason(self, ranker):
        sig = _make_signal(reason="ORB breakout")
        result = ranker.rank([sig])
        assert result[0].reason.startswith("ORB breakout")
        assert "[score=" in result[0].reason


# ===================================================================
# Score components
# ===================================================================

class TestScoreComponents:

    def test_regime_score_bull_favours_orb(self, ranker, bull_regime_probs):
        """ORB has high affinity in LOW_VOL_BULL; regime score should be elevated."""
        sig = _make_signal(strategy="ORB")
        result = ranker.rank([sig], regime_probs=bull_regime_probs)
        score_str = result[0].reason.split("[score=")[1].rstrip("]")
        score = float(score_str)
        assert score > 0.3  # not a great score but definitely not zero

    def test_regime_score_bear_penalises_orb(self, ranker, bear_regime_probs):
        """ORB has low affinity in HIGH_VOL_BEAR; score should be lower."""
        sig_bull = _make_signal(strategy="ORB", reason="bull")
        sig_bear = _make_signal(strategy="ORB", reason="bear")
        [r_bull] = ranker.rank([sig_bull], regime_probs={
            MarketRegimeState.LOW_VOL_BULL: 0.9,
            MarketRegimeState.HIGH_VOL_BULL: 0.05,
            MarketRegimeState.LOW_VOL_BEAR: 0.02,
            MarketRegimeState.HIGH_VOL_BEAR: 0.01,
            MarketRegimeState.MEAN_REVERTING: 0.02,
        })
        [r_bear] = ranker.rank([sig_bear], regime_probs=bear_regime_probs)
        score_bull = float(r_bull.reason.split("[score=")[1].rstrip("]"))
        score_bear = float(r_bear.reason.split("[score=")[1].rstrip("]"))
        assert score_bull > score_bear

    def test_win_rate_boosts_score(self, ranker, trade_history_df):
        """Signal for symbol/strategy with high win rate should score higher."""
        good = _make_signal(symbol="AAPL", strategy="ORB",
                            entry_price=150, take_profit=155, stop_loss=148)
        bad = _make_signal(symbol="MSFT", strategy="VWAP",
                           entry_price=400, take_profit=405, stop_loss=398)
        result = ranker.rank([bad, good], trade_history=trade_history_df)
        # AAPL/ORB has 70 % WR vs MSFT/VWAP 40 % WR
        assert result[0].symbol == "AAPL"

    def test_confidence_score_increases_with_rr(self, ranker):
        """Higher R:R ratio should produce higher confidence score."""
        low = _make_signal(entry_price=100, take_profit=101, stop_loss=99)   # 1:1
        high = _make_signal(entry_price=100, take_profit=105, stop_loss=99)  # 5:1
        scores = []
        for sig in [low, high]:
            s = ranker._confidence_score(sig)
            scores.append(s)
        assert scores[1] > scores[0]

    def test_confidence_zero_risk(self, ranker):
        """If stop_loss equals entry, confidence should be 0."""
        sig = _make_signal(entry_price=100, stop_loss=100, take_profit=105)
        assert ranker._confidence_score(sig) == 0.0

    def test_confidence_caps_at_1(self, ranker):
        """R:R >= 5 should cap confidence at 1.0."""
        sig = _make_signal(entry_price=100, take_profit=120, stop_loss=99)  # 20:1
        assert ranker._confidence_score(sig) == 1.0


# ===================================================================
# Trade history fallback
# ===================================================================

class TestTradeHistoryFallback:

    def test_empty_history_returns_defaults(self, ranker):
        sig = _make_signal()
        result = ranker.rank([sig], trade_history=pd.DataFrame())
        assert len(result) == 1
        assert "[score=" in result[0].reason

    def test_none_history_returns_defaults(self, ranker):
        sig = _make_signal()
        result = ranker.rank([sig], trade_history=None)
        assert len(result) == 1

    def test_insufficient_trades_falls_back(self, ranker):
        """With < 5 trades for symbol+strategy, falls back to strategy level."""
        df = pd.DataFrame([
            {"symbol": "AAPL", "strategy": "ORB", "pnl_pct": 0.01},
            {"symbol": "AAPL", "strategy": "ORB", "pnl_pct": 0.02},
            # Only 2 trades — should fallback
        ])
        wr, _, _ = ranker._trade_stats("AAPL", "ORB", df)
        # Falls all the way to defaults since strategy-level also has < 5
        assert wr == 0.50


# ===================================================================
# Expected value
# ===================================================================

class TestExpectedValue:

    def test_ev_positive_for_good_strategy(self, ranker, trade_history_df):
        sig = _make_signal(symbol="AAPL", strategy="ORB")
        ev = ranker.get_expected_value(sig, trade_history=trade_history_df)
        # 70% WR * 2% avg_win - 30% * 1% avg_loss = 0.014 - 0.003 = 0.011
        assert ev > 0

    def test_ev_with_no_history(self, ranker):
        sig = _make_signal()
        ev = ranker.get_expected_value(sig, trade_history=None)
        # Default: 0.5*0.01 - 0.5*0.01 = 0
        assert ev == 0.0

    def test_ev_negative_for_losing_strategy(self, ranker):
        """Strategy with low win rate and small wins should have negative EV."""
        rows = []
        for _ in range(2):
            rows.append({"symbol": "X", "strategy": "BAD", "pnl_pct": 0.005})
        for _ in range(8):
            rows.append({"symbol": "X", "strategy": "BAD", "pnl_pct": -0.02})
        df = pd.DataFrame(rows)
        sig = _make_signal(symbol="X", strategy="BAD")
        ev = ranker.get_expected_value(sig, trade_history=df)
        assert ev < 0


# ===================================================================
# Pair signal handling
# ===================================================================

class TestPairSignals:

    def test_pair_legs_stay_adjacent(self, ranker):
        """Both legs of a pair should be adjacent in the output."""
        leg1 = _make_signal(symbol="AAPL", strategy="KALMAN_PAIRS", side="buy",
                            entry_price=150, take_profit=155, stop_loss=148,
                            pair_id="AAPL-MSFT-001")
        leg2 = _make_signal(symbol="MSFT", strategy="KALMAN_PAIRS", side="sell",
                            entry_price=400, take_profit=395, stop_loss=404,
                            pair_id="AAPL-MSFT-001")
        single = _make_signal(symbol="NVDA", strategy="ORB",
                              entry_price=800, take_profit=900, stop_loss=750)

        result = ranker.rank([leg1, single, leg2])

        # Find the pair legs in the result
        pair_indices = [i for i, s in enumerate(result) if s.pair_id == "AAPL-MSFT-001"]
        assert len(pair_indices) == 2
        assert abs(pair_indices[0] - pair_indices[1]) == 1


# ===================================================================
# Unknown strategy
# ===================================================================

class TestUnknownStrategy:

    def test_unknown_strategy_gets_default_regime_score(self, ranker, bull_regime_probs):
        """A strategy not in STRATEGY_REGIME_AFFINITY should get 0.5 regime score."""
        sig = _make_signal(strategy="UNKNOWN_STRAT")
        score = ranker._regime_score("UNKNOWN_STRAT", bull_regime_probs)
        assert score == 0.5

    def test_unknown_strategy_still_ranks(self, ranker):
        sig = _make_signal(strategy="MYSTERY")
        result = ranker.rank([sig])
        assert len(result) == 1
        assert "[score=" in result[0].reason
