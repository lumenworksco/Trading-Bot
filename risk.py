"""Risk management — position sizing, circuit breaker, portfolio limits."""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import config

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
    status: str = "open"    # "open", "closed"
    order_id: str = ""
    time_stop: datetime | None = None  # for VWAP time stop


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
        self.starting_equity = equity
        self.current_equity = equity
        self.current_cash = cash
        self.day_pnl = 0.0
        self.circuit_breaker_active = False
        self.open_trades.clear()
        self.closed_trades.clear()
        self.signals_today = 0

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

    def can_open_trade(self) -> tuple[bool, str]:
        """Check if we can open a new trade. Returns (allowed, reason)."""
        if self.circuit_breaker_active:
            return False, "Circuit breaker active"

        if len(self.open_trades) >= config.MAX_POSITIONS:
            return False, f"Max positions ({config.MAX_POSITIONS}) reached"

        # Check total deployed capital
        deployed = sum(
            t.entry_price * t.qty for t in self.open_trades.values()
        )
        max_deploy = self.current_equity * config.MAX_PORTFOLIO_DEPLOY
        if deployed >= max_deploy:
            return False, f"Max portfolio deployment ({config.MAX_PORTFOLIO_DEPLOY:.0%}) reached"

        return True, ""

    def calculate_position_size(self, price: float, regime: str) -> int:
        """Calculate number of shares to buy based on sizing rules."""
        trade_amount = self.current_cash * config.TRADE_SIZE_PCT

        # Cut size in bearish regime
        if regime == "BEARISH":
            trade_amount *= (1 - config.BEARISH_SIZE_CUT)

        # Check we don't exceed max deployment
        deployed = sum(
            t.entry_price * t.qty for t in self.open_trades.values()
        )
        remaining_deploy = self.current_equity * config.MAX_PORTFOLIO_DEPLOY - deployed
        trade_amount = min(trade_amount, remaining_deploy)

        if trade_amount <= 0 or price <= 0:
            return 0

        qty = int(trade_amount / price)
        return max(qty, 0)

    def register_trade(self, trade: TradeRecord):
        """Register a new open trade."""
        self.open_trades[trade.symbol] = trade
        self.signals_today += 1
        logger.info(
            f"Trade opened: {trade.side.upper()} {trade.qty} {trade.symbol} "
            f"@ {trade.entry_price:.2f} ({trade.strategy}) "
            f"TP={trade.take_profit:.2f} SL={trade.stop_loss:.2f}"
        )

    def close_trade(self, symbol: str, exit_price: float, now: datetime):
        """Close a trade and record P&L."""
        if symbol not in self.open_trades:
            return

        trade = self.open_trades.pop(symbol)
        trade.exit_price = exit_price
        trade.exit_time = now
        trade.status = "closed"

        if trade.side == "buy":
            trade.pnl = (exit_price - trade.entry_price) * trade.qty
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.qty

        self.closed_trades.append(trade)
        logger.info(
            f"Trade closed: {trade.symbol} ({trade.strategy}) "
            f"P&L=${trade.pnl:+.2f} ({trade.pnl / (trade.entry_price * trade.qty):.1%})"
        )

    def get_day_summary(self) -> dict:
        """Generate end-of-day summary stats."""
        all_trades = self.closed_trades
        if not all_trades:
            return {"trades": 0}

        winners = [t for t in all_trades if t.pnl > 0]
        losers = [t for t in all_trades if t.pnl <= 0]
        orb_trades = [t for t in all_trades if t.strategy == "ORB"]
        vwap_trades = [t for t in all_trades if t.strategy == "VWAP"]
        orb_winners = [t for t in orb_trades if t.pnl > 0]
        vwap_winners = [t for t in vwap_trades if t.pnl > 0]

        total_pnl = sum(t.pnl for t in all_trades)
        best = max(all_trades, key=lambda t: t.pnl)
        worst = min(all_trades, key=lambda t: t.pnl)

        return {
            "trades": len(all_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(all_trades) if all_trades else 0,
            "total_pnl": total_pnl,
            "pnl_pct": total_pnl / self.starting_equity if self.starting_equity else 0,
            "best_trade": f"{best.symbol} {best.strategy} ${best.pnl:+.0f}",
            "worst_trade": f"{worst.symbol} {worst.strategy} ${worst.pnl:+.0f}",
            "orb_win_rate": f"{len(orb_winners)}/{len(orb_trades)}" if orb_trades else "N/A",
            "vwap_win_rate": f"{len(vwap_winners)}/{len(vwap_trades)}" if vwap_trades else "N/A",
        }

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
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
        """Restore state from persistence."""
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
            )
        logger.info(f"Restored {len(self.open_trades)} open trades from state")
