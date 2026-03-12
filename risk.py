"""Risk management — position sizing, circuit breaker, portfolio limits."""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import config
import database

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    symbol: str
    strategy: str
    side: str
    entry_price: float
    entry_time: datetime
    qty: int
    take_profit: float
    stop_loss: float
    pnl: float = 0.0
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str = ""            # V2: 'take_profit', 'stop_loss', 'time_stop', 'eod_close', 'max_hold'
    status: str = "open"             # "open", "closed"
    order_id: str = ""
    time_stop: datetime | None = None
    hold_type: str = "day"           # V2: "day" or "swing" (multi-day)
    max_hold_date: datetime | None = None  # V2: for momentum max hold


@dataclass
class RiskManager:
    starting_equity: float = 0.0
    current_equity: float = 0.0
    current_cash: float = 0.0
    day_pnl: float = 0.0
    circuit_breaker_active: bool = False
    open_trades: dict = field(default_factory=dict)     # symbol -> TradeRecord
    closed_trades: list = field(default_factory=list)    # today's closed trades
    signals_today: int = 0

    def reset_daily(self, equity: float, cash: float):
        """Reset daily state. Preserves swing (multi-day) trades."""
        self.starting_equity = equity
        self.current_equity = equity
        self.current_cash = cash
        self.day_pnl = 0.0
        self.circuit_breaker_active = False
        self.closed_trades.clear()
        self.signals_today = 0

        # Preserve swing trades, clear day trades
        day_trades = [s for s, t in self.open_trades.items() if t.hold_type == "day"]
        for symbol in day_trades:
            self.open_trades.pop(symbol)

    def update_equity(self, equity: float, cash: float):
        self.current_equity = equity
        self.current_cash = cash
        if self.starting_equity > 0:
            self.day_pnl = (equity - self.starting_equity) / self.starting_equity

    def check_circuit_breaker(self) -> bool:
        """Check if daily loss limit hit. Returns True if trading should halt."""
        if self.day_pnl <= config.DAILY_LOSS_HALT:
            if not self.circuit_breaker_active:
                logger.warning(
                    f"CIRCUIT BREAKER ACTIVATED: Day P&L {self.day_pnl:.2%} "
                    f"hit limit of {config.DAILY_LOSS_HALT:.2%}"
                )
            self.circuit_breaker_active = True
            return True
        return False

    def can_open_trade(self, strategy: str = "") -> tuple[bool, str]:
        """Check if we can open a new trade. Returns (allowed, reason)."""
        if self.circuit_breaker_active:
            return False, "Circuit breaker active"

        if len(self.open_trades) >= config.MAX_POSITIONS:
            return False, f"Max positions ({config.MAX_POSITIONS}) reached"

        # Check momentum-specific limit
        if strategy == "MOMENTUM":
            momentum_count = sum(1 for t in self.open_trades.values() if t.strategy == "MOMENTUM")
            if momentum_count >= config.MAX_MOMENTUM_POSITIONS:
                return False, f"Max momentum positions ({config.MAX_MOMENTUM_POSITIONS}) reached"

        # Check total deployed capital
        deployed = sum(
            t.entry_price * t.qty for t in self.open_trades.values()
        )
        max_deploy = self.current_equity * config.MAX_PORTFOLIO_DEPLOY
        if deployed >= max_deploy:
            return False, f"Max portfolio deployment ({config.MAX_PORTFOLIO_DEPLOY:.0%}) reached"

        return True, ""

    def calculate_position_size(self, entry_price: float, stop_price: float, regime: str) -> int:
        """ATR-based position sizing: risk exactly 1% of portfolio per trade.

        Args:
            entry_price: Expected entry price
            stop_price: Stop loss price
            regime: Market regime ('BULLISH', 'BEARISH', 'UNKNOWN')
        """
        # Risk per trade = 1% of portfolio
        risk_per_trade = self.current_equity * config.RISK_PER_TRADE_PCT

        # Distance to stop in dollars per share
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0 or entry_price <= 0:
            return 0

        # Shares = how many to risk exactly 1%
        shares = risk_per_trade / risk_per_share
        position_value = shares * entry_price

        # Hard caps
        max_position = self.current_equity * config.MAX_POSITION_PCT
        position_value = max(config.MIN_POSITION_VALUE, min(position_value, max_position))

        # Cut size in bearish regime
        if regime == "BEARISH":
            position_value *= (1 - config.BEARISH_SIZE_CUT)

        # Check we don't exceed max deployment
        deployed = sum(t.entry_price * t.qty for t in self.open_trades.values())
        remaining_deploy = self.current_equity * config.MAX_PORTFOLIO_DEPLOY - deployed
        position_value = min(position_value, remaining_deploy)

        if position_value <= 0:
            return 0

        qty = int(position_value / entry_price)
        return max(qty, 0)

    def register_trade(self, trade: TradeRecord):
        """Register a new open trade."""
        self.open_trades[trade.symbol] = trade
        self.signals_today += 1
        logger.info(
            f"Trade opened: {trade.side.upper()} {trade.qty} {trade.symbol} "
            f"@ {trade.entry_price:.2f} ({trade.strategy}/{trade.hold_type}) "
            f"TP={trade.take_profit:.2f} SL={trade.stop_loss:.2f}"
        )

    def close_trade(self, symbol: str, exit_price: float, now: datetime,
                    exit_reason: str = ""):
        """Close a trade, record P&L, and log to database."""
        if symbol not in self.open_trades:
            return

        trade = self.open_trades.pop(symbol)
        trade.exit_price = exit_price
        trade.exit_time = now
        trade.status = "closed"
        trade.exit_reason = exit_reason

        if trade.side == "buy":
            trade.pnl = (exit_price - trade.entry_price) * trade.qty
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.qty

        pnl_pct = trade.pnl / (trade.entry_price * trade.qty) if trade.entry_price * trade.qty > 0 else 0

        self.closed_trades.append(trade)
        logger.info(
            f"Trade closed: {trade.symbol} ({trade.strategy}) "
            f"P&L=${trade.pnl:+.2f} ({pnl_pct:.1%}) reason={exit_reason}"
        )

        # Log to database
        try:
            database.log_trade(
                symbol=trade.symbol,
                strategy=trade.strategy,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                qty=trade.qty,
                entry_time=trade.entry_time,
                exit_time=now,
                exit_reason=exit_reason,
                pnl=trade.pnl,
                pnl_pct=pnl_pct,
            )
        except Exception as e:
            logger.error(f"Failed to log trade to DB: {e}")

    def get_day_summary(self) -> dict:
        """Generate end-of-day summary stats."""
        all_trades = self.closed_trades
        if not all_trades:
            return {"trades": 0}

        winners = [t for t in all_trades if t.pnl > 0]
        losers = [t for t in all_trades if t.pnl <= 0]

        # Per-strategy breakdown
        strategies = {}
        for t in all_trades:
            s = t.strategy
            if s not in strategies:
                strategies[s] = {"total": 0, "winners": 0}
            strategies[s]["total"] += 1
            if t.pnl > 0:
                strategies[s]["winners"] += 1

        total_pnl = sum(t.pnl for t in all_trades)
        best = max(all_trades, key=lambda t: t.pnl)
        worst = min(all_trades, key=lambda t: t.pnl)

        result = {
            "trades": len(all_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(all_trades) if all_trades else 0,
            "total_pnl": total_pnl,
            "pnl_pct": total_pnl / self.starting_equity if self.starting_equity else 0,
            "best_trade": f"{best.symbol} {best.strategy} ${best.pnl:+.0f}",
            "worst_trade": f"{worst.symbol} {worst.strategy} ${worst.pnl:+.0f}",
        }

        # Add per-strategy win rates
        for strat, data in strategies.items():
            result[f"{strat.lower()}_win_rate"] = f"{data['winners']}/{data['total']}"

        return result

    def load_from_db(self):
        """Restore open positions from database."""
        try:
            rows = database.load_open_positions()
            for row in rows:
                self.open_trades[row["symbol"]] = TradeRecord(
                    symbol=row["symbol"],
                    strategy=row["strategy"],
                    side=row["side"],
                    entry_price=row["entry_price"],
                    entry_time=datetime.fromisoformat(row["entry_time"]),
                    qty=int(row["qty"]),
                    take_profit=row["take_profit"],
                    stop_loss=row["stop_loss"],
                    order_id=row.get("alpaca_order_id", ""),
                    hold_type=row.get("hold_type", "day"),
                    time_stop=datetime.fromisoformat(row["time_stop"]) if row.get("time_stop") else None,
                    max_hold_date=datetime.fromisoformat(row["max_hold_date"]) if row.get("max_hold_date") else None,
                )
            logger.info(f"Restored {len(self.open_trades)} open trades from database")
        except Exception as e:
            logger.error(f"Failed to load positions from DB: {e}")

    # --- Legacy serialization (kept for migration compatibility) ---

    def to_dict(self) -> dict:
        """Serialize state for persistence (legacy)."""
        return {
            "starting_equity": self.starting_equity,
            "day_pnl": self.day_pnl,
            "circuit_breaker_active": self.circuit_breaker_active,
            "signals_today": self.signals_today,
            "open_trades": {
                symbol: {
                    "symbol": t.symbol,
                    "strategy": t.strategy,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "entry_time": t.entry_time.isoformat(),
                    "qty": t.qty,
                    "take_profit": t.take_profit,
                    "stop_loss": t.stop_loss,
                    "order_id": t.order_id,
                    "time_stop": t.time_stop.isoformat() if t.time_stop else None,
                    "hold_type": t.hold_type,
                }
                for symbol, t in self.open_trades.items()
            },
            "closed_trades": [
                {
                    "symbol": t.symbol,
                    "strategy": t.strategy,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "qty": t.qty,
                    "pnl": t.pnl,
                }
                for t in self.closed_trades
            ],
        }

    def load_from_dict(self, d: dict, now: datetime):
        """Restore state from JSON (legacy, for migration)."""
        self.starting_equity = d.get("starting_equity", 0)
        self.day_pnl = d.get("day_pnl", 0)
        self.circuit_breaker_active = d.get("circuit_breaker_active", False)
        self.signals_today = d.get("signals_today", 0)

        for symbol, td in d.get("open_trades", {}).items():
            self.open_trades[symbol] = TradeRecord(
                symbol=td["symbol"],
                strategy=td["strategy"],
                side=td["side"],
                entry_price=td["entry_price"],
                entry_time=datetime.fromisoformat(td["entry_time"]),
                qty=td["qty"],
                take_profit=td["take_profit"],
                stop_loss=td["stop_loss"],
                order_id=td.get("order_id", ""),
                time_stop=datetime.fromisoformat(td["time_stop"]) if td.get("time_stop") else None,
                hold_type=td.get("hold_type", "day"),
            )
        logger.info(f"Restored {len(self.open_trades)} open trades from state")
