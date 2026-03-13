"""Catalyst Momentum strategy — holds 1-5 days."""

import logging
from datetime import datetime, timedelta

import pandas_ta as ta
from alpaca.data.timeframe import TimeFrame

import config
from data import get_daily_bars, get_intraday_bars
from strategies.base import Signal

logger = logging.getLogger(__name__)


class MomentumStrategy:
    """Catalyst Momentum: trades post-catalyst continuation patterns.

    Holds 1-5 trading days (overnight). Only LONG.
    Max 1 momentum position at a time.
    Scanned once daily at 10:30 AM ET.
    """

    def __init__(self):
        self.scanned_today: bool = False
        self.triggered: set[str] = set()  # symbols already triggered (ever, until reset)

    def reset_daily(self):
        self.scanned_today = False

    def scan(self, symbols: list[str], now: datetime, regime_detector=None) -> list[Signal]:
        """Scan for catalyst momentum setups. Call once daily at 10:30 AM.

        Signal logic:
        1. Yesterday moved > +4% on volume > 2x 30-day average
        2. Today's price within 1.5% of yesterday's close (consolidation)
        3. After 11:00 AM, new intraday high above pause day high (1-hour chart)
        4. Stock above 20-day EMA
        5. SPY positive on the day
        """
        if self.scanned_today:
            return []

        self.scanned_today = True
        signals = []

        # Check SPY macro tailwind
        if regime_detector and not regime_detector.is_spy_positive_today():
            logger.info("Momentum: SPY negative today, skipping scan")
            return signals

        for symbol in symbols:
            if symbol in self.triggered:
                continue

            # Skip leveraged ETFs
            if symbol in config.LEVERAGED_ETFS:
                continue

            try:
                signal = self._check_symbol(symbol, now)
                if signal:
                    signals.append(signal)
                    self.triggered.add(symbol)
            except Exception as e:
                logger.warning(f"Momentum scan error for {symbol}: {e}")

        if signals:
            logger.info(f"Momentum: found {len(signals)} setups")

        return signals

    def _check_symbol(self, symbol: str, now: datetime) -> Signal | None:
        """Check a single symbol for momentum setup."""
        # Get recent daily bars (need ~35 days for 30-day avg volume + EMA)
        daily = get_daily_bars(symbol, days=40)
        if daily.empty or len(daily) < 32:
            return None

        yesterday = daily.iloc[-2]
        day_before = daily.iloc[-3]

        # --- Condition 1: Yesterday moved > +4% on high volume ---
        yesterday_move = (yesterday["close"] - day_before["close"]) / day_before["close"]
        if yesterday_move < config.MOMENTUM_MIN_MOVE_PCT:
            return None

        # Volume check: yesterday's volume > 2x 30-day average
        vol_window = daily["volume"].iloc[-32:-2]  # 30 days before yesterday
        avg_volume = vol_window.mean()
        if avg_volume == 0 or yesterday["volume"] < config.MOMENTUM_VOL_MULTIPLIER * avg_volume:
            return None

        # --- Condition 2: Today within 1.5% of yesterday's close (consolidation) ---
        today = daily.iloc[-1]
        today_vs_yesterday = abs(today["close"] - yesterday["close"]) / yesterday["close"]
        if today_vs_yesterday > config.MOMENTUM_CONSOLIDATION_PCT:
            return None

        # --- Condition 4: Stock above 20-day EMA ---
        ema = ta.ema(daily["close"], length=20)
        if ema is None or ema.empty:
            return None
        if today["close"] < ema.iloc[-1]:
            return None

        # --- Condition 3: After 11 AM, check for intraday breakout above pause day high ---
        # The pause day high is today's high so far
        pause_high = today["high"]

        # Get intraday 1-hour bars to check for breakout
        today_date = now.date()
        intraday_start = datetime(today_date.year, today_date.month, today_date.day, 9, 30, tzinfo=config.ET)
        bars = get_intraday_bars(symbol, TimeFrame.Hour, start=intraday_start, end=now)
        if bars.empty:
            return None

        # Check if any bar after 11 AM made a new high above pause_high
        breakout_found = False
        for idx, bar in bars.iterrows():
            bar_time = idx
            if hasattr(bar_time, 'hour') and bar_time.hour >= 11:
                if bar["high"] > pause_high:
                    breakout_found = True
                    break
            elif hasattr(bar_time, 'astimezone'):
                et_time = bar_time.astimezone(config.ET)
                if et_time.hour >= 11 and bar["high"] > pause_high:
                    breakout_found = True
                    break

        if not breakout_found:
            return None

        # --- Calculate entry/exit levels ---
        entry_price = round(pause_high + 0.01, 2)  # Just above pause high

        # Stop loss: below pause day low, max -2% from entry
        pause_low = today["low"]
        stop_from_low = pause_low - 0.01
        stop_from_pct = entry_price * (1 - config.MOMENTUM_MAX_STOP_PCT)
        stop_loss = round(max(stop_from_low, stop_from_pct), 2)  # Use tighter stop

        # Take profit: first target at +3% (will sell 50% here)
        take_profit = round(entry_price * (1 + config.MOMENTUM_TP1_PCT), 2)

        return Signal(
            symbol=symbol,
            strategy="MOMENTUM",
            side="buy",
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            reason=f"Catalyst +{yesterday_move:.1%} on {yesterday['volume']/avg_volume:.1f}x vol, consolidation breakout",
            hold_type="swing",
        )
