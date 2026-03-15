"""V8: On-Balance Volume (OBV) divergence detection.

Used as a confidence modifier for mean reversion signals.
Bullish divergence boosts BUY confidence; bearish divergence reduces it.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_obv(closes: pd.Series, volumes: pd.Series) -> list[float]:
    """Compute cumulative On-Balance Volume.

    Args:
        closes: Series of close prices
        volumes: Series of volumes

    Returns:
        List of OBV values
    """
    close_vals = closes.values.astype(float)
    vol_vals = volumes.values.astype(float)

    obv = [0.0]
    for i in range(1, len(close_vals)):
        if close_vals[i] > close_vals[i - 1]:
            obv.append(obv[-1] + vol_vals[i])
        elif close_vals[i] < close_vals[i - 1]:
            obv.append(obv[-1] - vol_vals[i])
        else:
            obv.append(obv[-1])
    return obv


def detect_obv_divergence(closes: pd.Series, volumes: pd.Series,
                          lookback: int = 20) -> str | None:
    """Detect OBV divergence with price.

    Bullish divergence: price making lower lows, OBV making higher lows
    Bearish divergence: price making higher highs, OBV making lower highs

    Args:
        closes: Series of close prices
        volumes: Series of volumes
        lookback: Number of bars to analyze

    Returns:
        'bullish', 'bearish', or None
    """
    if len(closes) < lookback + 5:
        return None

    try:
        close_vals = closes.values[-lookback:].astype(float)
        obv_vals = np.array(compute_obv(closes.iloc[-lookback:], volumes.iloc[-lookback:]))

        # Split into two halves for comparison
        mid = lookback // 2

        first_half_close = close_vals[:mid]
        second_half_close = close_vals[mid:]
        first_half_obv = obv_vals[:mid]
        second_half_obv = obv_vals[mid:]

        # Find lows and highs in each half
        first_close_low = np.min(first_half_close)
        second_close_low = np.min(second_half_close)
        first_close_high = np.max(first_half_close)
        second_close_high = np.max(second_half_close)

        first_obv_low = np.min(first_half_obv)
        second_obv_low = np.min(second_half_obv)
        first_obv_high = np.max(first_half_obv)
        second_obv_high = np.max(second_half_obv)

        # Bullish divergence: price lower low, OBV higher low
        if second_close_low < first_close_low and second_obv_low > first_obv_low:
            return "bullish"

        # Bearish divergence: price higher high, OBV lower high
        if second_close_high > first_close_high and second_obv_high < first_obv_high:
            return "bearish"

    except Exception as e:
        logger.debug(f"OBV divergence detection failed: {e}")

    return None
