"""Intraday Seasonality — scores each time window for favorability per strategy type.

Different strategies perform better at different times of day. For example,
mean-reversion strategies thrive during the low-volatility lunch lull, while
ORB only fires in the opening drive. This module provides per-strategy,
per-window multipliers that the signal pipeline uses to scale position sizes.
"""

from datetime import datetime, time
from typing import Optional

import config

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore


class IntradaySeasonality:
    """Scores each time window for favorability per strategy type."""

    WINDOWS = {
        "open_auction":  (time(9, 30), time(9, 45)),
        "opening_drive": (time(9, 45), time(10, 30)),
        "mid_morning":   (time(10, 30), time(11, 30)),
        "lunch_lull":    (time(11, 30), time(14, 0)),
        "afternoon":     (time(14, 0), time(15, 30)),
        "close_auction": (time(15, 30), time(16, 0)),
    }

    # Pre-configured strategy-window affinity scores
    STRATEGY_WINDOW_SCORES = {
        "STAT_MR":      {"open_auction": 0.3, "opening_drive": 0.7, "mid_morning": 1.0, "lunch_lull": 1.3, "afternoon": 1.0, "close_auction": 0.5},
        "VWAP":         {"open_auction": 0.3, "opening_drive": 0.7, "mid_morning": 1.0, "lunch_lull": 1.3, "afternoon": 1.0, "close_auction": 0.5},
        "KALMAN_PAIRS": {"open_auction": 0.5, "opening_drive": 0.8, "mid_morning": 1.0, "lunch_lull": 1.1, "afternoon": 1.0, "close_auction": 0.7},
        "ORB":          {"open_auction": 0.0, "opening_drive": 1.5, "mid_morning": 1.0, "lunch_lull": 0.0, "afternoon": 0.0, "close_auction": 0.0},
        "MICRO_MOM":    {"open_auction": 0.5, "opening_drive": 1.3, "mid_morning": 1.0, "lunch_lull": 0.5, "afternoon": 1.3, "close_auction": 0.8},
        "PEAD":         {"open_auction": 1.0, "opening_drive": 1.0, "mid_morning": 1.0, "lunch_lull": 1.0, "afternoon": 1.0, "close_auction": 1.0},
    }

    # Valid score range for clamping
    SCORE_MIN = 0.0
    SCORE_MAX = 1.5

    def __init__(self):
        self._adaptive_scores: Optional[dict] = None  # For future adaptive learning

    def get_current_window(self, now: datetime) -> Optional[str]:
        """Return the name of the current time window, or None if outside market hours.

        Parameters
        ----------
        now : datetime
            A timezone-aware datetime (should be in ET).

        Returns
        -------
        str or None
            Window name like ``"opening_drive"`` or ``None`` if *now* falls
            outside all defined windows.
        """
        t = now.time()
        for name, (start, end) in self.WINDOWS.items():
            if start <= t < end:
                return name
        return None

    def get_window_score(self, now: datetime, strategy: str) -> float:
        """Return the seasonality multiplier for *strategy* at time *now*.

        Returns
        -------
        float
            A value in [0.0, 1.5].  Returns ``1.0`` for unknown strategies
            or when *now* is outside market hours.

            If ``config.SEASONALITY_OPEN_AUCTION_BLOCK`` is ``True`` and the
            current window is ``"open_auction"``, returns ``0.0`` regardless
            of strategy.
        """
        if not getattr(config, "INTRADAY_SEASONALITY_ENABLED", True):
            return 1.0

        window = self.get_current_window(now)

        # Outside market hours -> neutral
        if window is None:
            return 1.0

        # Block open auction entirely when configured
        if window == "open_auction" and getattr(config, "SEASONALITY_OPEN_AUCTION_BLOCK", True):
            return 0.0

        # Check adaptive scores first (future feature)
        if self._adaptive_scores is not None:
            adaptive = self._adaptive_scores.get(strategy, {}).get(window)
            if adaptive is not None:
                return max(self.SCORE_MIN, min(self.SCORE_MAX, adaptive))

        # Lookup static table
        strategy_scores = self.STRATEGY_WINDOW_SCORES.get(strategy)
        if strategy_scores is None:
            return 1.0

        score = strategy_scores.get(window, 1.0)
        return max(self.SCORE_MIN, min(self.SCORE_MAX, score))

    def update_from_data(self, trade_history) -> None:
        """Adaptively learn seasonality from actual trading data.

        This is a placeholder for future implementation.  When
        ``config.SEASONALITY_ADAPTIVE_LEARNING`` is ``True`` and sufficient
        trade history is available, this method will compute per-strategy,
        per-window win-rate and average P&L adjustments.

        Parameters
        ----------
        trade_history : pandas.DataFrame
            Must contain columns: strategy, entry_time, pnl.
        """
        if not getattr(config, "SEASONALITY_ADAPTIVE_LEARNING", False):
            return
        # Future: compute window-level performance from trade_history
        # and populate self._adaptive_scores
        pass
