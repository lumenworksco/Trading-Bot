"""Opening Range Breakout (ORB) strategy."""

import logging
from datetime import datetime, timedelta

from alpaca.data.timeframe import TimeFrame

import config
from data import get_intraday_bars, get_daily_bars
from strategies.base import Signal, ORBRange

logger = logging.getLogger(__name__)


class ORBStrategy:
    """Opening Range Breakout strategy."""

    def __init__(self):
        self.ranges: dict[str, ORBRange] = {}
        self.triggered: set[str] = set()  # symbols already triggered today
        self.top_volume_symbols: list[str] = []
        self.ranges_recorded = False

    def reset_daily(self):
        self.ranges.clear()
        self.triggered.clear()
        self.top_volume_symbols.clear()
        self.ranges_recorded = False

    def record_opening_ranges(self, symbols: list[str], now: datetime):
        """Record ORB high/low for each symbol using 5-min bars from 9:30-10:00."""
        if self.ranges_recorded:
            return

        today = now.date()
        orb_start = datetime(today.year, today.month, today.day, 9, 30, tzinfo=config.ET)
        orb_end = datetime(today.year, today.month, today.day, 10, 0, tzinfo=config.ET)

        for symbol in symbols:
            try:
                bars = get_intraday_bars(symbol, TimeFrame.Minute, start=orb_start, end=orb_end)
                if bars.empty or len(bars) < 5:
                    continue

                orb_high = bars["high"].max()
                orb_low = bars["low"].min()
                orb_volume = bars["volume"].sum()

                # Get previous close for gap check
                daily = get_daily_bars(symbol, days=3)
                if daily.empty or len(daily) < 2:
                    continue
                prev_close = daily["close"].iloc[-2]

                self.ranges[symbol] = ORBRange(
                    high=orb_high,
                    low=orb_low,
                    volume=orb_volume,
                    prev_close=prev_close,
                )
            except Exception as e:
                logger.warning(f"Failed to record ORB for {symbol}: {e}")

        # Rank by ORB volume and pick top N
        ranked = sorted(self.ranges.items(), key=lambda x: x[1].volume, reverse=True)
        self.top_volume_symbols = [s for s, _ in ranked[:config.ORB_TOP_N_SYMBOLS]]
        self.ranges_recorded = True
        logger.info(f"ORB ranges recorded for {len(self.ranges)} symbols, top {len(self.top_volume_symbols)} selected")

    def scan(self, symbols: list[str], now: datetime, regime: str = "UNKNOWN") -> list[Signal]:
        """Scan for ORB breakout signals."""
        signals = []

        if not self.ranges_recorded:
            return signals

        for symbol in self.top_volume_symbols:
            if symbol in self.triggered:
                continue
            if symbol not in self.ranges:
                continue

            orb = self.ranges[symbol]

            # Gap filter
            today_open = orb.low  # approximate
            gap_pct = abs(today_open - orb.prev_close) / orb.prev_close
            if gap_pct > config.MAX_GAP_PCT:
                continue

            # Range quality filter
            orb_range = orb.high - orb.low
            range_pct = orb_range / ((orb.high + orb.low) / 2)
            if range_pct > config.MAX_ORB_RANGE_PCT:
                continue

            try:
                # Get recent 5-min bars to check for breakout
                lookback = now - timedelta(minutes=10)
                bars = get_intraday_bars(symbol, TimeFrame(5, "Min"), start=lookback, end=now)
                if bars.empty:
                    continue

                latest = bars.iloc[-1]

                # Volume check: current bar volume > 1.5x average
                avg_volume = orb.volume / 6  # 30 min / 5 min = 6 bars average
                if latest["volume"] < config.ORB_VOLUME_MULTIPLIER * avg_volume:
                    continue

                # LONG: close above ORB high
                if latest["close"] > orb.high:
                    entry_price = orb.high * (1 + config.ORB_ENTRY_SLIPPAGE)
                    take_profit = entry_price + config.ORB_TAKE_PROFIT_MULT * orb_range
                    stop_loss = entry_price - config.ORB_STOP_LOSS_MULT * orb_range

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="ORB",
                        side="buy",
                        entry_price=round(entry_price, 2),
                        take_profit=round(take_profit, 2),
                        stop_loss=round(stop_loss, 2),
                        reason=f"Breakout above ORB high {orb.high:.2f}, vol={latest['volume']:.0f}",
                        hold_type="day",
                    ))
                    self.triggered.add(symbol)

                # V3 SHORT: close below ORB low (bearish/choppy regime only)
                elif (
                    config.ALLOW_SHORT
                    and latest["close"] < orb.low
                    and regime in ("BEARISH", "UNKNOWN")
                    and symbol not in config.NO_SHORT_SYMBOLS
                ):
                    # Confirm stock is actually down on the day (>0.5%)
                    day_change = (latest["close"] - orb.prev_close) / orb.prev_close
                    if day_change < -0.005:
                        entry_price = orb.low * (1 - config.ORB_ENTRY_SLIPPAGE)
                        take_profit = entry_price - config.ORB_TAKE_PROFIT_MULT * orb_range
                        stop_loss = entry_price + config.ORB_STOP_LOSS_MULT * orb_range

                        signals.append(Signal(
                            symbol=symbol,
                            strategy="ORB",
                            side="sell",
                            entry_price=round(entry_price, 2),
                            take_profit=round(take_profit, 2),
                            stop_loss=round(stop_loss, 2),
                            reason=f"Breakdown below ORB low {orb.low:.2f}, vol={latest['volume']:.0f}",
                            hold_type="day",
                        ))
                        self.triggered.add(symbol)

            except Exception as e:
                logger.warning(f"ORB scan error for {symbol}: {e}")

        return signals
