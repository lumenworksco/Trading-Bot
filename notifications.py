"""V3: Telegram notifications — trade alerts, daily summaries, warnings."""

import logging
from datetime import datetime

import config

logger = logging.getLogger(__name__)


def _send_telegram(message: str):
    """Send a message via Telegram bot API. Synchronous (fire-and-forget)."""
    if not config.TELEGRAM_ENABLED:
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return

    try:
        import httpx
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
            })
            if resp.status_code != 200:
                logger.warning(f"Telegram send failed: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")


def notify_trade_opened(symbol: str, side: str, strategy: str,
                        entry_price: float, qty: int,
                        take_profit: float, stop_loss: float):
    """Notify when a new position is opened."""
    arrow = "\U0001f4c8" if side == "buy" else "\U0001f4c9"
    _send_telegram(
        f"{arrow} <b>OPENED</b> {side.upper()} {symbol}\n"
        f"Strategy: {strategy}\n"
        f"Entry: ${entry_price:.2f} | Size: {qty} shares\n"
        f"TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}"
    )


def notify_trade_closed(symbol: str, pnl: float, pnl_pct: float,
                        exit_reason: str, hold_time: str = ""):
    """Notify when a position is closed."""
    icon = "\u2705" if pnl > 0 else "\u274c"
    _send_telegram(
        f"{icon} <b>CLOSED</b> {symbol}\n"
        f"P&L: ${pnl:+.2f} ({pnl_pct:+.2%})\n"
        f"Reason: {exit_reason}"
        + (f" | Hold: {hold_time}" if hold_time else "")
    )


def notify_circuit_breaker(day_pnl_pct: float):
    """Notify when circuit breaker triggers."""
    _send_telegram(
        f"\U0001f6a8 <b>CIRCUIT BREAKER TRIGGERED</b>\n"
        f"Daily P&L: {day_pnl_pct:.2%}\n"
        f"No new trades for remainder of day."
    )


def notify_daily_summary(day_pnl: float, day_pnl_pct: float,
                         n_trades: int, win_rate: float,
                         best_trade: str, worst_trade: str):
    """Notify with end-of-day summary."""
    _send_telegram(
        f"\U0001f4ca <b>DAILY SUMMARY</b>\n"
        f"P&L: ${day_pnl:+.2f} ({day_pnl_pct:+.2%})\n"
        f"Trades: {n_trades} | Win rate: {win_rate:.0%}\n"
        f"Best: {best_trade} | Worst: {worst_trade}"
    )


def notify_drawdown_warning(drawdown_pct: float):
    """Notify when portfolio drawdown exceeds threshold."""
    _send_telegram(
        f"\u26a0\ufe0f <b>DRAWDOWN WARNING</b>\n"
        f"Portfolio down {drawdown_pct:.2%} from peak.\n"
        f"Position sizing reduced."
    )


def notify_ml_retrain(results: dict):
    """Notify about ML model retraining results."""
    lines = ["\U0001f9e0 <b>ML MODEL RETRAINED</b>"]
    for strategy, metrics in results.items():
        status = "\u2705" if metrics["active"] else "\u274c"
        lines.append(
            f"{status} {strategy}: precision={metrics['precision']:.1%}, "
            f"n={metrics['train_samples']}"
        )
    _send_telegram("\n".join(lines))


def notify_optimization(strategy: str, old_sharpe: float, new_sharpe: float,
                        params: dict):
    """Notify when strategy parameters are optimized."""
    _send_telegram(
        f"\U0001f527 <b>{strategy} PARAMS UPDATED</b>\n"
        f"Sharpe: {old_sharpe:.2f} \u2192 {new_sharpe:.2f}\n"
        f"New params: {params}"
    )
