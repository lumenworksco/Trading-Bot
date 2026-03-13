"""Hurst exponent estimation via R/S (Rescaled Range) analysis."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def hurst_exponent(prices: pd.Series, max_lag: int = 100) -> float:
    """
    Compute Hurst exponent using R/S analysis.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending

    Args:
        prices: Price series (at least 50 observations recommended)
        max_lag: Maximum lag for R/S computation

    Returns:
        Hurst exponent (float). Returns 0.5 on failure (assume random walk).
    """
    if len(prices) < 20:
        return 0.5

    try:
        ts = prices.dropna().values
        if len(ts) < 20:
            return 0.5

        max_lag = min(max_lag, len(ts) // 2)
        lags = range(2, max_lag)

        # Compute tau (standard deviation of lagged differences)
        tau = []
        for lag in lags:
            diffs = ts[lag:] - ts[:-lag]
            std = np.std(diffs)
            if std > 0:
                tau.append(std)
            else:
                tau.append(1e-10)

        if len(tau) < 5:
            return 0.5

        # Hurst exponent = slope of log-log plot
        log_lags = np.log(list(lags)[:len(tau)])
        log_tau = np.log(tau)

        poly = np.polyfit(log_lags, log_tau, 1)
        hurst = poly[0]

        # Clamp to reasonable range
        return float(max(0.0, min(1.0, hurst)))

    except Exception as e:
        logger.warning(f"Hurst computation failed: {e}")
        return 0.5
