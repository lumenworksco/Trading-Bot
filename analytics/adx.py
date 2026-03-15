"""V8: ADX (Average Directional Index) trend strength indicator.

Used as pre-trade filter:
- ORB breakouts only when ADX > 25 (trending market)
- VWAP mean reversion only when ADX < 20 (ranging market)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                period: int = 14) -> float | None:
    """Compute the Average Directional Index (Wilder's ADX).

    Args:
        highs: Series of high prices
        lows: Series of low prices
        closes: Series of close prices
        period: ADX period (default 14)

    Returns:
        Current ADX value (0-100), or None if insufficient data
    """
    if len(highs) < period * 2 + 1:
        return None

    try:
        high = highs.values.astype(float)
        low = lows.values.astype(float)
        close = closes.values.astype(float)

        n = len(close)

        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Wilder's smoothing (EMA-like with alpha = 1/period)
        def wilder_smooth(values, period):
            smoothed = np.zeros(len(values))
            smoothed[period] = np.sum(values[1:period + 1])
            for i in range(period + 1, len(values)):
                smoothed[i] = smoothed[i - 1] - smoothed[i - 1] / period + values[i]
            return smoothed

        atr = wilder_smooth(tr, period)
        plus_di_smooth = wilder_smooth(plus_dm, period)
        minus_di_smooth = wilder_smooth(minus_dm, period)

        # +DI and -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * plus_di_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_di_smooth[i] / atr[i]

        # DX
        dx = np.zeros(n)
        for i in range(period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        # ADX = smoothed DX
        adx = np.zeros(n)
        start = period * 2
        if start >= n:
            return None
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        result = adx[-1]
        return float(result) if not np.isnan(result) else None

    except Exception as e:
        logger.debug(f"ADX computation failed: {e}")
        return None
