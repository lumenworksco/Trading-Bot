"""SQLite database — replaces state.json for persistence and adds trade/signal logging."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.DB_FILE, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def init_db():
    """Create all tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            qty REAL,
            entry_time TEXT,
            exit_time TEXT,
            exit_reason TEXT,
            pnl REAL,
            pnl_pct REAL
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            acted_on INTEGER DEFAULT 0,
            skip_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS open_positions (
            symbol TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            qty REAL,
            entry_time TEXT,
            take_profit REAL,
            stop_loss REAL,
            alpaca_order_id TEXT,
            hold_type TEXT DEFAULT 'day',
            time_stop TEXT,
            max_hold_date TEXT
        );

        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date TEXT PRIMARY KEY,
            portfolio_value REAL,
            cash REAL,
            day_pnl REAL,
            day_pnl_pct REAL,
            total_trades INTEGER,
            win_rate REAL,
            sharpe_rolling REAL
        );

        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            strategy TEXT,
            total_return REAL,
            annualized_return REAL,
            sharpe_ratio REAL,
            win_rate REAL,
            profit_factor REAL,
            max_drawdown REAL,
            total_trades INTEGER,
            avg_hold_minutes REAL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
        CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
        CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
    """)
    conn.commit()
    logger.info("Database initialized")


# --- Trade Logging ---

def log_trade(symbol: str, strategy: str, side: str, entry_price: float,
              exit_price: float, qty: float, entry_time: datetime,
              exit_time: datetime, exit_reason: str, pnl: float, pnl_pct: float):
    """Log a completed trade."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO trades (symbol, strategy, side, entry_price, exit_price,
           qty, entry_time, exit_time, exit_reason, pnl, pnl_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, strategy, side, entry_price, exit_price, qty,
         entry_time.isoformat(), exit_time.isoformat(), exit_reason, pnl, pnl_pct),
    )
    conn.commit()


# --- Signal Logging ---

def log_signal(timestamp: datetime, symbol: str, strategy: str,
               signal_type: str, acted_on: bool, skip_reason: str = ""):
    """Log a signal (whether or not it was acted on)."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO signals (timestamp, symbol, strategy, signal_type,
           acted_on, skip_reason) VALUES (?, ?, ?, ?, ?, ?)""",
        (timestamp.isoformat(), symbol, strategy, signal_type,
         1 if acted_on else 0, skip_reason),
    )
    conn.commit()


# --- Open Positions (replaces state.json) ---

def save_open_positions(open_trades: dict):
    """Replace all open_positions rows with current state."""
    conn = _get_conn()
    conn.execute("DELETE FROM open_positions")
    for symbol, trade in open_trades.items():
        conn.execute(
            """INSERT INTO open_positions (symbol, strategy, side, entry_price,
               qty, entry_time, take_profit, stop_loss, alpaca_order_id,
               hold_type, time_stop, max_hold_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, trade.strategy, trade.side, trade.entry_price,
             trade.qty, trade.entry_time.isoformat(), trade.take_profit,
             trade.stop_loss, trade.order_id,
             getattr(trade, 'hold_type', 'day'),
             trade.time_stop.isoformat() if trade.time_stop else None,
             trade.max_hold_date.isoformat() if getattr(trade, 'max_hold_date', None) else None),
        )
    conn.commit()


def load_open_positions() -> list[dict]:
    """Load open positions from DB. Returns list of dicts."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM open_positions").fetchall()
    return [dict(row) for row in rows]


# --- Daily Snapshots ---

def save_daily_snapshot(date: str, portfolio_value: float, cash: float,
                        day_pnl: float, day_pnl_pct: float,
                        total_trades: int, win_rate: float, sharpe_rolling: float):
    """Insert or replace daily snapshot."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO daily_snapshots
           (date, portfolio_value, cash, day_pnl, day_pnl_pct,
            total_trades, win_rate, sharpe_rolling)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (date, portfolio_value, cash, day_pnl, day_pnl_pct,
         total_trades, win_rate, sharpe_rolling),
    )
    conn.commit()


# --- Analytics Queries ---

def get_recent_trades(days: int = 7) -> list[dict]:
    """Get trades from the last N days."""
    conn = _get_conn()
    cutoff = (datetime.now(config.ET) - __import__('datetime').timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM trades WHERE exit_time >= ? ORDER BY exit_time DESC",
        (cutoff,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_all_trades() -> list[dict]:
    """Get all trades ever."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM trades ORDER BY exit_time DESC").fetchall()
    return [dict(row) for row in rows]


def get_daily_snapshots(days: int = 30) -> list[dict]:
    """Get daily snapshots for the last N days."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_daily_pnl_series(days: int = 30) -> list[float]:
    """Get list of daily P&L percentages for analytics."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT day_pnl_pct FROM daily_snapshots ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [row["day_pnl_pct"] for row in reversed(rows)]


def get_portfolio_values(days: int = 30) -> list[float]:
    """Get list of portfolio values for drawdown calculation."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT portfolio_value FROM daily_snapshots ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [row["portfolio_value"] for row in reversed(rows)]


def get_signal_stats_today() -> dict:
    """Get today's signal statistics."""
    conn = _get_conn()
    today = datetime.now(config.ET).strftime("%Y-%m-%d")
    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE timestamp LIKE ?",
        (f"{today}%",),
    ).fetchone()["cnt"]
    acted = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE timestamp LIKE ? AND acted_on = 1",
        (f"{today}%",),
    ).fetchone()["cnt"]
    return {"total": total, "acted": acted, "skipped": total - acted}


# --- Backtest Results ---

def save_backtest_result(run_date: str, strategy: str, total_return: float,
                         annualized_return: float, sharpe_ratio: float,
                         win_rate: float, profit_factor: float,
                         max_drawdown: float, total_trades: int,
                         avg_hold_minutes: float):
    """Save backtest results."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO backtest_results
           (run_date, strategy, total_return, annualized_return, sharpe_ratio,
            win_rate, profit_factor, max_drawdown, total_trades, avg_hold_minutes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_date, strategy, total_return, annualized_return, sharpe_ratio,
         win_rate, profit_factor, max_drawdown, total_trades, avg_hold_minutes),
    )
    conn.commit()


# --- Migration ---

def migrate_from_json():
    """One-time migration from state.json to SQLite."""
    json_path = Path(config.STATE_FILE)
    if not json_path.exists():
        return

    try:
        data = json.loads(json_path.read_text())
        # Migrate open trades
        for symbol, td in data.get("open_trades", {}).items():
            conn = _get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO open_positions
                   (symbol, strategy, side, entry_price, qty, entry_time,
                    take_profit, stop_loss, alpaca_order_id, hold_type, time_stop)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, td["strategy"], td["side"], td["entry_price"],
                 td["qty"], td["entry_time"], td["take_profit"],
                 td["stop_loss"], td.get("order_id", ""), "day",
                 td.get("time_stop")),
            )
        conn = _get_conn()
        conn.commit()

        # Rename json file so migration doesn't run again
        backup = json_path.with_suffix(".json.bak")
        json_path.rename(backup)
        logger.info(f"Migrated state.json to SQLite, backup at {backup}")

    except Exception as e:
        logger.error(f"Migration from state.json failed: {e}")
