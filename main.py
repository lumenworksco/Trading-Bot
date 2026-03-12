"""Entry point — main loop, startup checks, state persistence."""

import json
import logging
import sys
import time as time_mod
from datetime import datetime, time, timedelta
from pathlib import Path

from rich.console import Console
from rich.live import Live

import config
from data import verify_connectivity, verify_data_feed, get_account, get_clock, get_positions
from strategies import MarketRegime, ORBStrategy, VWAPStrategy, Signal
from risk import RiskManager, TradeRecord
from execution import (
    submit_bracket_order,
    close_position,
    close_orb_positions,
    check_vwap_time_stops,
    cancel_all_open_orders,
)
from dashboard import (
    build_dashboard,
    print_day_summary,
    print_startup_info,
    console,
)

# --- Logging setup ---
# File handler always active; stream handler removed once Live dashboard starts
_file_handler = logging.FileHandler(config.LOG_FILE)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)


def now_et() -> datetime:
    return datetime.now(config.ET)


def is_market_hours(t: time) -> bool:
    return config.MARKET_OPEN <= t <= config.MARKET_CLOSE


def is_trading_hours(t: time) -> bool:
    return config.TRADING_START <= t <= config.ORB_EXIT_TIME


def is_orb_recording_period(t: time) -> bool:
    return config.MARKET_OPEN <= t < config.ORB_END


def save_state(risk: RiskManager):
    """Save state to JSON file."""
    try:
        state = risk.to_dict()
        state["saved_at"] = now_et().isoformat()
        Path(config.STATE_FILE).write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def load_state(risk: RiskManager):
    """Load state from JSON file if it exists."""
    try:
        path = Path(config.STATE_FILE)
        if path.exists():
            data = json.loads(path.read_text())
            risk.load_from_dict(data, now_et())
            logger.info(f"State loaded from {config.STATE_FILE}")
    except Exception as e:
        logger.error(f"Failed to load state: {e}")


def startup_checks() -> dict:
    """Run all startup checks. Exit on failure."""
    console.print("\n[bold]Running startup checks...[/bold]\n")

    # 1. Verify API connectivity
    try:
        info = verify_connectivity()
        print_startup_info(info)
    except Exception as e:
        console.print(f"[bold red]FATAL: Cannot connect to Alpaca API: {e}[/bold red]")
        console.print("Check ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        sys.exit(1)

    # 2. Check market status
    if not info["market_open"]:
        next_open = info.get("next_open", "unknown")
        console.print(f"[yellow]Market is closed. Next open: {next_open}[/yellow]")
        console.print("[yellow]Bot will wait for market open...[/yellow]")

    # 3. Verify data feed
    console.print("Verifying data feed...")
    try:
        if not verify_data_feed("SPY"):
            console.print("[bold red]FATAL: Cannot fetch market data. Check API permissions.[/bold red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]FATAL: Data feed error: {e}[/bold red]")
        sys.exit(1)
    console.print("[green]Data feed verified.[/green]")

    # 4. Print symbol list
    console.print(f"\n[bold]Symbol universe:[/bold] {len(config.SYMBOLS)} symbols")
    console.print(", ".join(config.SYMBOLS[:10]) + f"... and {len(config.SYMBOLS) - 10} more\n")

    return info


def process_signals(
    signals: list[Signal],
    risk: RiskManager,
    regime: str,
    now: datetime,
):
    """Process signals: check risk, size, and submit orders."""
    for signal in signals:
        # Skip if already in this symbol
        if signal.symbol in risk.open_trades:
            continue

        # Check risk limits
        allowed, reason = risk.can_open_trade()
        if not allowed:
            logger.info(f"Trade blocked for {signal.symbol}: {reason}")
            continue

        # Calculate position size
        qty = risk.calculate_position_size(signal.entry_price, regime)
        if qty <= 0:
            logger.info(f"Position size 0 for {signal.symbol}, skipping")
            continue

        # Submit bracket order
        order_id = submit_bracket_order(signal, qty)
        if order_id is None:
            logger.error(f"Failed to submit order for {signal.symbol}, skipping (no naked entry)")
            continue

        # Register the trade
        time_stop = None
        if signal.strategy == "VWAP":
            time_stop = now + timedelta(minutes=config.VWAP_TIME_STOP_MINUTES)

        trade = TradeRecord(
            symbol=signal.symbol,
            strategy=signal.strategy,
            side=signal.side,
            entry_price=signal.entry_price,
            entry_time=now,
            qty=qty,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            order_id=order_id,
            time_stop=time_stop,
        )
        risk.register_trade(trade)


def sync_positions_with_broker(risk: RiskManager, now: datetime):
    """Sync open trades with actual broker positions to detect fills/stops."""
    try:
        broker_positions = {p.symbol: p for p in get_positions()}
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return

    # Check if any of our tracked trades have been closed by the broker
    for symbol in list(risk.open_trades.keys()):
        if symbol not in broker_positions:
            # Position was closed (TP or SL hit)
            trade = risk.open_trades[symbol]
            # We don't know the exact exit price, estimate from TP/SL
            # In production you'd check the order fill price
            risk.close_trade(symbol, trade.entry_price, now)  # Approximate
            logger.info(f"Position {symbol} no longer at broker — marking closed")

    # Update equity from account
    try:
        account = get_account()
        risk.update_equity(float(account.equity), float(account.cash))
    except Exception as e:
        logger.error(f"Failed to update account: {e}")


def main():
    """Main entry point."""
    console.print("[bold cyan]Starting Algo Trading Bot...[/bold cyan]\n")

    # Startup checks
    info = startup_checks()

    # Initialize components
    regime_detector = MarketRegime()
    orb = ORBStrategy()
    vwap = VWAPStrategy()
    risk = RiskManager()

    # Initialize risk with account info
    risk.reset_daily(info["equity"], info["cash"])

    # Load persisted state
    load_state(risk)

    start_time = now_et()
    last_scan = None
    last_state_save = now_et()
    last_day = now_et().date()
    eod_summary_printed = False

    console.print("[bold green]Bot is running. Press Ctrl+C to stop.[/bold green]\n")

    # Stop logging to terminal — dashboard takes over the screen
    logging.getLogger().removeHandler(_stream_handler)

    try:
        with Live(
            build_dashboard(risk, regime_detector.regime, start_time, now_et(), last_scan, len(config.SYMBOLS)),
            console=console,
            refresh_per_second=0.2,  # update every 5 seconds
            transient=False,
        ) as live:
            while True:
                current = now_et()
                current_time = current.time()

                # Daily reset
                if current.date() != last_day:
                    logger.info("New trading day — resetting state")
                    orb.reset_daily()
                    vwap.reset_daily()
                    try:
                        account = get_account()
                        risk.reset_daily(float(account.equity), float(account.cash))
                    except Exception as e:
                        logger.error(f"Failed to reset daily: {e}")
                    last_day = current.date()
                    eod_summary_printed = False

                # Update regime
                regime = regime_detector.update(current)

                # Market hours logic
                if is_market_hours(current_time):

                    # ORB recording period (9:30-10:00)
                    if is_orb_recording_period(current_time):
                        if not orb.ranges_recorded and current_time >= time(9, 55):
                            # Record at 9:55 to ensure we have most of the range
                            orb.record_opening_ranges(config.SYMBOLS, current)

                    # Trading hours (10:00-15:45)
                    if is_trading_hours(current_time):
                        signals = []

                        # Run strategies based on regime
                        if regime in ("BULLISH", "UNKNOWN"):
                            orb_signals = orb.scan(config.SYMBOLS, current)
                            signals.extend(orb_signals)

                        # VWAP runs in all regimes
                        vwap_signals = vwap.scan(config.SYMBOLS, current, regime)
                        signals.extend(vwap_signals)

                        # Process signals
                        if signals:
                            process_signals(signals, risk, regime, current)

                        # Check VWAP time stops
                        expired = check_vwap_time_stops(risk.open_trades, current)
                        for symbol in expired:
                            risk.close_trade(symbol, risk.open_trades[symbol].entry_price if symbol in risk.open_trades else 0, current)

                    # ORB exit time (15:45)
                    if current_time >= config.ORB_EXIT_TIME:
                        closed = close_orb_positions(risk.open_trades, current)
                        for symbol in closed:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current)

                    # Sync with broker
                    sync_positions_with_broker(risk, current)

                    # Check circuit breaker
                    risk.check_circuit_breaker()

                    last_scan = current

                # EOD summary at 16:15
                if current_time >= config.EOD_SUMMARY_TIME and not eod_summary_printed:
                    summary = risk.get_day_summary()
                    print_day_summary(summary)
                    logger.info(f"Day summary: {summary}")
                    eod_summary_printed = True

                # Save state periodically
                if (current - last_state_save).total_seconds() >= config.STATE_SAVE_INTERVAL_SEC:
                    save_state(risk)
                    last_state_save = current

                # Update dashboard
                live.update(
                    build_dashboard(risk, regime, start_time, current, last_scan, len(config.SYMBOLS))
                )

                # Sleep until next scan
                time_mod.sleep(config.SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        save_state(risk)
        console.print("[green]State saved. Bot stopped.[/green]")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        save_state(risk)
        console.print(f"[bold red]Bot crashed: {e}[/bold red]")
        console.print("[yellow]State saved. Restart with: python main.py[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
