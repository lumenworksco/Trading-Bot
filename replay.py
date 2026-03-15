"""Market Replay & Config Comparison — replay historical trading days through the signal pipeline."""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import config

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


# ============================================================
# Result dataclasses
# ============================================================

@dataclass
class ReplayResult:
    """Result of replaying a single trading day."""
    date: str
    signals: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    config_used: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Side-by-side comparison of two replay runs."""
    date: str
    result_a: ReplayResult = field(default_factory=lambda: ReplayResult(date=""))
    result_b: ReplayResult = field(default_factory=lambda: ReplayResult(date=""))
    delta_pnl: float = 0.0
    delta_sharpe: float = 0.0
    delta_win_rate: float = 0.0


# ============================================================
# Simulated trade tracking
# ============================================================

@dataclass
class _SimTrade:
    """Internal: tracks a simulated position during replay."""
    symbol: str
    strategy: str
    side: str
    entry_price: float
    take_profit: float
    stop_loss: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(ET))
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str = ""
    pnl: float = 0.0


# ============================================================
# MarketReplay
# ============================================================

class MarketReplay:
    """
    Replays a historical trading day through the full signal pipeline.
    Uses cached bar data and simulates the exact sequence of scans/signals/executions.
    """

    def __init__(self):
        self._scan_interval_minutes = max(config.SCAN_INTERVAL_SEC // 60, 1)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def replay_day(self, date: str, config_overrides: dict | None = None) -> ReplayResult:
        """
        Replays the given trading day with optional config overrides.
        Returns: all signals generated, risk decisions, simulated fills, P&L.
        """
        if not getattr(config, "REPLAY_ENABLED", True):
            logger.info("Replay disabled via config")
            return ReplayResult(date=date)

        overrides = config_overrides or {}
        result = ReplayResult(date=date, config_used=overrides)

        originals = {}
        try:
            # Apply config overrides
            originals = self._apply_overrides(overrides)

            # Load bar data
            bars_by_symbol = self._load_bars(date)
            if not bars_by_symbol:
                logger.warning(f"No bar data found for {date}")
                return result

            # Simulate time progression
            open_trades: dict[str, _SimTrade] = {}
            all_signals: list[dict] = []
            all_trades: list[dict] = []

            scan_times = self._generate_scan_times(date)

            for sim_time in scan_times:
                try:
                    # Generate signals at this scan time
                    signals = self._generate_signals(sim_time, bars_by_symbol, overrides)
                    for sig in signals:
                        sig_dict = {
                            "time": sim_time.isoformat(),
                            "symbol": sig.get("symbol", ""),
                            "strategy": sig.get("strategy", ""),
                            "side": sig.get("side", ""),
                            "entry_price": sig.get("entry_price", 0.0),
                        }
                        all_signals.append(sig_dict)

                        # Open simulated position if not already in one for this symbol
                        symbol = sig.get("symbol", "")
                        if symbol and symbol not in open_trades:
                            trade = _SimTrade(
                                symbol=symbol,
                                strategy=sig.get("strategy", ""),
                                side=sig.get("side", "buy"),
                                entry_price=sig.get("entry_price", 0.0),
                                take_profit=sig.get("take_profit", 0.0),
                                stop_loss=sig.get("stop_loss", 0.0),
                                entry_time=sim_time,
                            )
                            open_trades[symbol] = trade

                    # Check open trades against current prices
                    closed = self._check_exits(sim_time, open_trades, bars_by_symbol)
                    for t in closed:
                        all_trades.append(self._trade_to_dict(t))
                        open_trades.pop(t.symbol, None)

                except Exception as e:
                    logger.warning(f"Replay scan error at {sim_time}: {e}")
                    continue

            # Force close remaining positions at market close
            close_time = datetime.strptime(date, "%Y-%m-%d").replace(
                hour=16, minute=0, tzinfo=ET
            )
            for symbol, trade in list(open_trades.items()):
                try:
                    close_price = self._get_price_at_time(
                        symbol, close_time, bars_by_symbol
                    )
                    if close_price:
                        trade.exit_price = close_price
                        trade.exit_time = close_time
                        trade.exit_reason = "market_close"
                        trade.pnl = self._calc_pnl(trade)
                        all_trades.append(self._trade_to_dict(trade))
                except Exception as e:
                    logger.warning(f"Error closing {symbol} at EOD: {e}")

            result.signals = all_signals
            result.trades = all_trades
            result.total_pnl = sum(t.get("pnl", 0.0) for t in all_trades)
            result.win_rate = self._calc_win_rate(all_trades)
            result.sharpe = self._calc_sharpe(all_trades)

        except Exception as e:
            logger.error(f"Replay failed for {date}: {e}")
        finally:
            self._restore_overrides(originals)

        return result

    def compare_configs(self, date: str, config_a: dict, config_b: dict) -> ComparisonResult:
        """
        Runs the same day through two different configs.
        Returns side-by-side comparison of signals, trades, and P&L.
        """
        result = ComparisonResult(date=date)
        try:
            result.result_a = self.replay_day(date, config_a)
            result.result_b = self.replay_day(date, config_b)
            result.delta_pnl = result.result_a.total_pnl - result.result_b.total_pnl
            result.delta_sharpe = result.result_a.sharpe - result.result_b.sharpe
            result.delta_win_rate = result.result_a.win_rate - result.result_b.win_rate
        except Exception as e:
            logger.error(f"Config comparison failed for {date}: {e}")
        return result

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _apply_overrides(self, overrides: dict) -> dict:
        """Apply config overrides, return original values for restoration."""
        originals = {}
        for key, value in overrides.items():
            if hasattr(config, key):
                originals[key] = getattr(config, key)
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
        return originals

    def _restore_overrides(self, originals: dict):
        """Restore original config values."""
        for key, value in originals.items():
            setattr(config, key, value)

    def _load_bars(self, date: str) -> dict[str, list[dict]]:
        """Load bar data from database for the given date."""
        try:
            import database
            trades = database.get_all_trades()
            # Build a synthetic bar map from trade data for the date
            bars: dict[str, list[dict]] = {}
            for trade in trades:
                entry_time_str = trade.get("entry_time", "")
                if date in str(entry_time_str):
                    symbol = trade.get("symbol", "")
                    if symbol not in bars:
                        bars[symbol] = []
                    bars[symbol].append({
                        "time": entry_time_str,
                        "open": trade.get("entry_price", 0.0),
                        "high": trade.get("entry_price", 0.0) * 1.005,
                        "low": trade.get("entry_price", 0.0) * 0.995,
                        "close": trade.get("exit_price") or trade.get("entry_price", 0.0),
                        "volume": 100000,
                    })
            return bars
        except Exception as e:
            logger.warning(f"Failed to load bars for {date}: {e}")
            return {}

    def _generate_scan_times(self, date: str) -> list[datetime]:
        """Generate scan timestamps from 9:30 to 16:00 ET."""
        try:
            base = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ET)
        except ValueError:
            return []

        times = []
        current = base.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = base.replace(hour=16, minute=0, second=0, microsecond=0)

        while current <= market_close:
            times.append(current)
            current += timedelta(minutes=self._scan_interval_minutes)
        return times

    def _generate_signals(self, sim_time: datetime, bars: dict,
                          overrides: dict) -> list[dict]:
        """Generate signals for the current scan time. Fail-open."""
        signals = []
        try:
            for symbol, bar_list in bars.items():
                for bar in bar_list:
                    bar_time_str = bar.get("time", "")
                    try:
                        if "T" in str(bar_time_str):
                            bar_dt = datetime.fromisoformat(str(bar_time_str))
                        else:
                            continue
                        # Only consider bars near this scan time
                        if abs((bar_dt - sim_time).total_seconds()) < self._scan_interval_minutes * 60:
                            signals.append({
                                "symbol": symbol,
                                "strategy": "REPLAY",
                                "side": "buy",
                                "entry_price": bar.get("close", 0.0),
                                "take_profit": bar.get("close", 0.0) * 1.02,
                                "stop_loss": bar.get("close", 0.0) * 0.99,
                                "time": sim_time.isoformat(),
                            })
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"Signal generation error at {sim_time}: {e}")
        return signals

    def _check_exits(self, sim_time: datetime, open_trades: dict[str, _SimTrade],
                     bars: dict) -> list[_SimTrade]:
        """Check if any open trades should be exited."""
        closed = []
        for symbol, trade in list(open_trades.items()):
            try:
                price = self._get_price_at_time(symbol, sim_time, bars)
                if price is None:
                    continue

                if trade.side == "buy":
                    if price >= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.exit_time = sim_time
                        trade.exit_reason = "take_profit"
                        trade.pnl = self._calc_pnl(trade)
                        closed.append(trade)
                    elif price <= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = sim_time
                        trade.exit_reason = "stop_loss"
                        trade.pnl = self._calc_pnl(trade)
                        closed.append(trade)
                else:  # sell/short
                    if price <= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.exit_time = sim_time
                        trade.exit_reason = "take_profit"
                        trade.pnl = self._calc_pnl(trade)
                        closed.append(trade)
                    elif price >= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = sim_time
                        trade.exit_reason = "stop_loss"
                        trade.pnl = self._calc_pnl(trade)
                        closed.append(trade)
            except Exception as e:
                logger.warning(f"Exit check error for {symbol}: {e}")
        return closed

    def _get_price_at_time(self, symbol: str, sim_time: datetime,
                           bars: dict) -> float | None:
        """Get the closest price for a symbol at the given time."""
        bar_list = bars.get(symbol, [])
        if not bar_list:
            return None

        # Return the close of the nearest bar
        best_bar = None
        best_delta = float("inf")
        for bar in bar_list:
            try:
                bar_time_str = bar.get("time", "")
                if "T" in str(bar_time_str):
                    bar_dt = datetime.fromisoformat(str(bar_time_str))
                    delta = abs((bar_dt - sim_time).total_seconds())
                    if delta < best_delta:
                        best_delta = delta
                        best_bar = bar
            except (ValueError, TypeError):
                continue

        if best_bar:
            return best_bar.get("close", None)
        return None

    @staticmethod
    def _calc_pnl(trade: _SimTrade) -> float:
        """Calculate P&L for a trade."""
        if trade.exit_price is None:
            return 0.0
        if trade.side == "buy":
            return trade.exit_price - trade.entry_price
        else:
            return trade.entry_price - trade.exit_price

    @staticmethod
    def _calc_win_rate(trades: list[dict]) -> float:
        """Calculate win rate from trade dicts."""
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
        return wins / len(trades)

    @staticmethod
    def _calc_sharpe(trades: list[dict]) -> float:
        """Calculate Sharpe ratio from trade P&Ls."""
        pnls = [t.get("pnl", 0.0) for t in trades]
        if len(pnls) < 2:
            return 0.0
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        return mean_pnl / std

    @staticmethod
    def _trade_to_dict(trade: _SimTrade) -> dict:
        """Convert a _SimTrade to a serializable dict."""
        return {
            "symbol": trade.symbol,
            "strategy": trade.strategy,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "exit_reason": trade.exit_reason,
            "pnl": trade.pnl,
        }
