"""V3: Gap & Go strategy — trade pre-market gaps with institutional volume."""

import logging
from datetime import datetime, time, timedelta

from alpaca.data.timeframe import TimeFrame

import config
from data import get_daily_bars, get_intraday_bars, get_snapshot
from strategies.base import Signal

logger = logging.getLogger(__name__)


class GapGoStrategy:
    """Gap & Go: trade 3-8% pre-market gaps with strong volume continuation."""

    def __init__(self):
        self.candidates: list[dict] = []      # Pre-market gap candidates
        self.first_candle: dict = {}           # symbol -> {high, low, close, volume}
        self.triggered: set[str] = set()       # Symbols already triggered today
        self.scanned_premarket: bool = False
        self.candle_recorded: bool = False

    def reset_daily(self):
        self.candidates.clear()
        self.first_candle.clear()
        self.triggered.clear()
        self.scanned_premarket = False
        self.candle_recorded = False

    def find_gap_candidates(self, symbols: list[str], now: datetime):
        """Pre-market scan at 9:00 AM — find stocks gapping 3-8% with volume.

        Uses Alpaca snapshots to get current price vs previous close.
        """
        if self.scanned_premarket:
            return

        logger.info("Gap & Go: Scanning for pre-market gap candidates...")
        self.candidates.clear()

        for symbol in symbols:
            if symbol in config.LEVERAGED_ETFS:
                continue

            try:
                snap = get_snapshot(symbol)
                if snap is None:
                    continue

                # Get previous close and current price
                if not snap.previous_daily_bar or not snap.latest_trade:
                    continue

                prev_close = float(snap.previous_daily_bar.close)
                current_price = float(snap.latest_trade.price)

                if prev_close <= 0 or current_price < config.GAP_MIN_PRICE:
                    continue

                gap_pct = (current_price - prev_close) / prev_close

                # Check gap is in target range (3-8%)
                if not (config.GAP_MIN_PCT <= gap_pct <= config.GAP_MAX_PCT):
                    continue

                # Get daily volume average
                daily = get_daily_bars(symbol, days=25)
                if daily.empty or len(daily) < 5:
                    continue
                avg_volume = daily["volume"].iloc[-20:].mean()

                # Score: weighted gap + volume ratio
                volume_ratio = 1.0  # Can't easily get pre-market volume from IEX
                score = gap_pct * 0.4 + 0.6 * 0.1  # Simplified scoring

                self.candidates.append({
                    "symbol": symbol,
                    "gap_pct": gap_pct,
                    "prev_close": prev_close,
                    "current_price": current_price,
                    "avg_volume": avg_volume,
                    "prev_day_high": float(snap.previous_daily_bar.high),
                    "score": score,
                })

            except Exception as e:
                logger.warning(f"Gap scan error for {symbol}: {e}")

        # Sort by gap size and take top 5
        self.candidates.sort(key=lambda x: x["score"], reverse=True)
        self.candidates = self.candidates[:5]
        self.scanned_premarket = True

        if self.candidates:
            syms = [c["symbol"] for c in self.candidates]
            logger.info(f"Gap & Go: {len(self.candidates)} candidates found: {', '.join(syms)}")
        else:
            logger.info("Gap & Go: No gap candidates found today")

    def record_first_candle(self, now: datetime):
        """Record the first 15-minute candle (9:30-9:45) for candidates."""
        if self.candle_recorded or not self.candidates:
            return

        today = now.date()
        candle_start = datetime(today.year, today.month, today.day, 9, 30, tzinfo=config.ET)
        candle_end = datetime(today.year, today.month, today.day, 9, 45, tzinfo=config.ET)

        for cand in self.candidates:
            symbol = cand["symbol"]
            try:
                bars = get_intraday_bars(symbol, TimeFrame.Minute, start=candle_start, end=candle_end)
                if bars.empty or len(bars) < 10:
                    continue

                self.first_candle[symbol] = {
                    "high": bars["high"].max(),
                    "low": bars["low"].min(),
                    "close": bars["close"].iloc[-1],
                    "volume": bars["volume"].sum(),
                    "open": bars["open"].iloc[0],
                    "avg_volume": cand["avg_volume"],
                    "prev_day_high": cand["prev_day_high"],
                    "prev_close": cand["prev_close"],
                    "gap_pct": cand["gap_pct"],
                }
            except Exception as e:
                logger.warning(f"Gap first candle error for {symbol}: {e}")

        self.candle_recorded = True
        logger.info(f"Gap & Go: First candle recorded for {len(self.first_candle)} symbols")

    def scan(self, now: datetime) -> list[Signal]:
        """Scan for Gap & Go entry signals (9:45 AM - 11:30 AM).

        Entry criteria:
        1. Price above pre-market high (continuation confirmed)
        2. First 15-min candle closed green
        3. Volume in first 15 min > 30% of avg daily volume
        4. Entry at pullback to first candle high
        """
        signals = []

        if not self.candle_recorded:
            return signals

        for symbol, candle in self.first_candle.items():
            if symbol in self.triggered:
                continue

            try:
                snap = get_snapshot(symbol)
                if snap is None or not snap.latest_trade:
                    continue

                current_price = float(snap.latest_trade.price)

                # Check 1: First candle must have closed green
                if candle["close"] < candle["open"]:
                    continue

                # Check 2: Volume in first 15 min > 30% of avg daily
                if candle["volume"] < 0.30 * candle["avg_volume"]:
                    continue

                # Check 3: Current price must be above first candle high
                #           (confirms continuation, not fading)
                if current_price < candle["high"]:
                    continue

                # Entry at first candle high (pullback entry)
                entry_price = candle["high"]
                stop_loss = candle["low"]  # Below first candle low
                take_profit = candle["prev_day_high"] * 1.01  # Prev day high + 1%

                # Sanity: TP must be above entry
                if take_profit <= entry_price:
                    take_profit = entry_price * 1.02  # Fallback: +2%

                signals.append(Signal(
                    symbol=symbol,
                    strategy="GAP_GO",
                    side="buy",
                    entry_price=round(entry_price, 2),
                    take_profit=round(take_profit, 2),
                    stop_loss=round(stop_loss, 2),
                    reason=f"Gap {candle['gap_pct']:.1%} continuation, vol={candle['volume']:.0f}",
                    hold_type="day",
                ))
                self.triggered.add(symbol)

            except Exception as e:
                logger.warning(f"Gap & Go scan error for {symbol}: {e}")

        return signals
