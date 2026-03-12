"""Trading strategies — ORB and VWAP Mean Reversion."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
from alpaca.data.timeframe import TimeFrame

import config
from data import get_intraday_bars, get_daily_bars, get_snapshots

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    symbol: str
    strategy: str          # "ORB" or "VWAP"
    side: str              # "buy" or "sell"
    entry_price: float
    take_profit: float
    stop_loss: float
    reason: str = ""


@dataclass
class ORBRange:
    high: float
    low: float
    volume: float          # total volume during ORB period
    prev_close: float      # yesterday's close for gap check


@dataclass
class VWAPState:
    vwap: float = 0.0
    upper_band: float = 0.0
    lower_band: float = 0.0
    cumulative_volume: float = 0.0
    cumulative_vp: float = 0.0    # volume * price
    cumulative_vp2: float = 0.0   # volume * price^2


class MarketRegime:
    """Simple regime detection based on SPY vs 20-day EMA."""

    def __init__(self):
        self.regime: str = "UNKNOWN"
        self.last_check: datetime | None = None

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

            latest_close = df["close"].iloc[-1]
            latest_ema = ema.iloc[-1]

            self.regime = "BULLISH" if latest_close > latest_ema else "BEARISH"
            self.last_check = now
            logger.info(f"Market regime: {self.regime} (SPY={latest_close:.2f}, EMA={latest_ema:.2f})")

        except Exception as e:
            logger.error(f"Regime check failed: {e}")
            if self.regime == "UNKNOWN":
                self.regime = "BULLISH"

        return self.regime


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

    def scan(self, symbols: list[str], now: datetime) -> list[Signal]:
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
            today_open = orb.low  # approximate — actual open is first bar's open
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

                # Need close above ORB high
                if latest["close"] <= orb.high:
                    continue

                # Volume check: current bar volume > 1.5x average
                avg_volume = orb.volume / 6  # 30 min / 5 min = 6 bars average
                if latest["volume"] < config.ORB_VOLUME_MULTIPLIER * avg_volume:
                    continue

                # Signal! Calculate levels
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
                ))
                self.triggered.add(symbol)

            except Exception as e:
                logger.warning(f"ORB scan error for {symbol}: {e}")

        return signals


class VWAPStrategy:
    """VWAP Mean Reversion strategy."""

    def __init__(self):
        self.states: dict[str, VWAPState] = {}
        self.triggered: dict[str, datetime] = {}  # symbol -> last trigger time (cooldown)
        self.daily_moves: dict[str, float] = {}    # symbol -> day's move % for trend filter

    def reset_daily(self):
        self.states.clear()
        self.triggered.clear()
        self.daily_moves.clear()

    def _compute_vwap(self, bars: pd.DataFrame) -> tuple[float, float, float] | None:
        """Compute VWAP and bands from intraday bars."""
        if bars.empty:
            return None

        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
        cum_vol = bars["volume"].cumsum()
        cum_vp = (typical_price * bars["volume"]).cumsum()

        if cum_vol.iloc[-1] == 0:
            return None

        vwap = cum_vp.iloc[-1] / cum_vol.iloc[-1]

        # Standard deviation of typical price weighted by volume
        cum_vp2 = (typical_price**2 * bars["volume"]).cumsum()
        variance = cum_vp2.iloc[-1] / cum_vol.iloc[-1] - vwap**2
        std_dev = np.sqrt(max(variance, 0))

        upper = vwap + config.VWAP_BAND_STD * std_dev
        lower = vwap - config.VWAP_BAND_STD * std_dev

        return vwap, upper, lower

    def scan(self, symbols: list[str], now: datetime, regime: str) -> list[Signal]:
        """Scan for VWAP mean reversion signals."""
        signals = []
        today = now.date()
        market_open = datetime(today.year, today.month, today.day, 9, 30, tzinfo=config.ET)

        for symbol in symbols:
            # Cooldown: don't re-trigger within 5 minutes
            if symbol in self.triggered:
                if (now - self.triggered[symbol]).total_seconds() < 300:
                    continue

            try:
                # Get all 1-min bars from open until now
                bars = get_intraday_bars(symbol, TimeFrame.Minute, start=market_open, end=now)
                if bars.empty or len(bars) < 15:
                    continue

                # Trend filter: skip if stock moved > 3% today
                first_price = bars["open"].iloc[0]
                last_price = bars["close"].iloc[-1]
                day_move = abs(last_price - first_price) / first_price
                self.daily_moves[symbol] = day_move

                if day_move > config.MAX_INTRADAY_MOVE_PCT:
                    continue

                # Compute VWAP + bands
                result = self._compute_vwap(bars)
                if result is None:
                    continue
                vwap, upper, lower = result

                # Compute RSI(14)
                rsi_series = ta.rsi(bars["close"], length=14)
                if rsi_series is None or rsi_series.empty:
                    continue
                rsi = rsi_series.iloc[-1]

                prev_bar = bars.iloc[-2]
                curr_bar = bars.iloc[-1]

                # BUY signal: price touched lower band and bounced back above
                if prev_bar["low"] <= lower and curr_bar["close"] > lower and rsi < config.VWAP_RSI_OVERSOLD:
                    std_dev = (upper - vwap) / config.VWAP_BAND_STD
                    stop_loss = lower - config.VWAP_STOP_EXTENSION * std_dev
                    signals.append(Signal(
                        symbol=symbol,
                        strategy="VWAP",
                        side="buy",
                        entry_price=round(curr_bar["close"], 2),
                        take_profit=round(vwap, 2),
                        stop_loss=round(stop_loss, 2),
                        reason=f"VWAP bounce at {lower:.2f}, RSI={rsi:.1f}",
                    ))
                    self.triggered[symbol] = now

                # SELL/SHORT signal: price touched upper band and dropped below
                elif (
                    config.ALLOW_SHORT
                    and prev_bar["high"] >= upper
                    and curr_bar["close"] < upper
                    and rsi > config.VWAP_RSI_OVERBOUGHT
                ):
                    std_dev = (upper - vwap) / config.VWAP_BAND_STD
                    stop_loss = upper + config.VWAP_STOP_EXTENSION * std_dev
                    signals.append(Signal(
                        symbol=symbol,
                        strategy="VWAP",
                        side="sell",
                        entry_price=round(curr_bar["close"], 2),
                        take_profit=round(vwap, 2),
                        stop_loss=round(stop_loss, 2),
                        reason=f"VWAP rejection at {upper:.2f}, RSI={rsi:.1f}",
                    ))
                    self.triggered[symbol] = now

            except Exception as e:
                logger.warning(f"VWAP scan error for {symbol}: {e}")

        return signals
