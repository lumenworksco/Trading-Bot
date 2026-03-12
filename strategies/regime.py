"""Market regime detection based on SPY vs 20-day EMA."""

import logging
from datetime import datetime

import pandas_ta as ta

import config
from data import get_daily_bars

logger = logging.getLogger(__name__)


class MarketRegime:
    """Simple regime detection based on SPY vs 20-day EMA."""

    def __init__(self):
        self.regime: str = "UNKNOWN"
        self.last_check: datetime | None = None
        self.spy_price: float = 0.0
        self.spy_ema: float = 0.0

    def update(self, now: datetime) -> str:
        if (
            self.last_check is not None
            and (now - self.last_check).total_seconds() < config.REGIME_CHECK_INTERVAL_MIN * 60
        ):
            return self.regime

        try:
            df = get_daily_bars("SPY", days=config.REGIME_EMA_PERIOD + 5)
            if df.empty or len(df) < config.REGIME_EMA_PERIOD:
                logger.warning("Not enough SPY data for regime check, defaulting to BULLISH")
                self.regime = "BULLISH"
                self.last_check = now
                return self.regime

            ema = ta.ema(df["close"], length=config.REGIME_EMA_PERIOD)
            if ema is None or ema.empty:
                self.regime = "BULLISH"
                self.last_check = now
                return self.regime

            self.spy_price = df["close"].iloc[-1]
            self.spy_ema = ema.iloc[-1]

            self.regime = "BULLISH" if self.spy_price > self.spy_ema else "BEARISH"
            self.last_check = now
            logger.info(f"Market regime: {self.regime} (SPY={self.spy_price:.2f}, EMA={self.spy_ema:.2f})")

        except Exception as e:
            logger.error(f"Regime check failed: {e}")
            if self.regime == "UNKNOWN":
                self.regime = "BULLISH"

        return self.regime

    def is_spy_positive_today(self) -> bool:
        """Check if SPY is positive on the day (for momentum filter)."""
        try:
            df = get_daily_bars("SPY", days=2)
            if df.empty or len(df) < 2:
                return False
            return df["close"].iloc[-1] > df["close"].iloc[-2]
        except Exception:
            return False
