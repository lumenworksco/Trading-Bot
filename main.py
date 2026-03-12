"""Entry point V2 — main loop, --backtest flag, filters, momentum, SQLite persistence."""

import argparse
import json
import logging
import sys
import time as time_mod
from datetime import datetime, time, timedelta
from pathlib import Path

from rich.console import Console
from rich.live import Live

import config
import database
import analytics as analytics_mod
from data import verify_connectivity, verify_data_feed, get_account, get_clock, get_positions
from strategies import MarketRegime, ORBStrategy, VWAPStrategy, MomentumStrategy, Signal
from risk import RiskManager, TradeRecord
from execution import (
    submit_bracket_order,
    close_position,
    close_orb_positions,
    check_vwap_time_stops,
    check_momentum_max_hold,
    cancel_all_open_orders,
)
from dashboard import (
    build_dashboard,
    print_day_summary,
    print_startup_info,
    console,
)
from earnings import load_earnings_cache, has_earnings_soon, get_excluded_count
from correlation import load_correlation_cache, is_too_correlated

# --- Logging setup ---
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
    console.print(f"\n[bold]Symbol universe:[/bold] {len(config.SYMBOLS)} symbols ({len(config.CORE_SYMBOLS)} core + {len(config.SYMBOLS) - len(config.CORE_SYMBOLS)} extended)")
    console.print(", ".join(config.SYMBOLS[:10]) + f"... and {len(config.SYMBOLS) - 10} more")
    console.print(f"[dim]Leveraged ETFs (VWAP only): {', '.join(sorted(config.LEVERAGED_ETFS))}[/dim]\n")

    return info


def process_signals(
    signals: list[Signal],
    risk: RiskManager,
    regime: str,
    now: datetime,
):
    """Process signals: check filters, risk, size, and submit orders."""
    for signal in signals:
        skip_reason = ""

        # Skip if already in this symbol
        if signal.symbol in risk.open_trades:
            skip_reason = "already_in_position"
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            continue

        # Earnings filter
        if has_earnings_soon(signal.symbol):
            skip_reason = "earnings_soon"
            logger.info(f"Signal skipped for {signal.symbol}: earnings soon")
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            continue

        # Correlation filter
        open_symbols = list(risk.open_trades.keys())
        if open_symbols and is_too_correlated(signal.symbol, open_symbols):
            skip_reason = "high_correlation"
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            continue

        # Check risk limits
        allowed, reason = risk.can_open_trade(strategy=signal.strategy)
        if not allowed:
            skip_reason = reason
            logger.info(f"Trade blocked for {signal.symbol}: {reason}")
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            continue

        # Calculate position size (V2: ATR-based)
        qty = risk.calculate_position_size(signal.entry_price, signal.stop_loss, regime)
        if qty <= 0:
            skip_reason = "position_size_zero"
            logger.info(f"Position size 0 for {signal.symbol}, skipping")
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            continue

        # Submit bracket order
        order_id = submit_bracket_order(signal, qty)
        if order_id is None:
            skip_reason = "order_failed"
            logger.error(f"Failed to submit order for {signal.symbol}, skipping (no naked entry)")
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            continue

        # Register the trade
        time_stop = None
        max_hold_date = None
        hold_type = getattr(signal, 'hold_type', 'day')

        if signal.strategy == "VWAP":
            time_stop = now + timedelta(minutes=config.VWAP_TIME_STOP_MINUTES)
        elif signal.strategy == "MOMENTUM":
            max_hold_date = now + timedelta(days=config.MOMENTUM_MAX_HOLD_DAYS)

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
            hold_type=hold_type,
            max_hold_date=max_hold_date,
        )
        risk.register_trade(trade)

        # Log signal as acted on
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, True, "")


def sync_positions_with_broker(risk: RiskManager, now: datetime):
    """Sync open trades with actual broker positions to detect fills/stops."""
    try:
        broker_positions = {p.symbol: p for p in get_positions()}
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return

    for symbol in list(risk.open_trades.keys()):
        if symbol not in broker_positions:
            trade = risk.open_trades[symbol]
            # Estimate exit: if trade P&L would be positive at TP, assume TP hit
            risk.close_trade(symbol, trade.entry_price, now, exit_reason="broker_sync")
            logger.info(f"Position {symbol} no longer at broker — marking closed")

    try:
        account = get_account()
        risk.update_equity(float(account.equity), float(account.cash))
    except Exception as e:
        logger.error(f"Failed to update account: {e}")


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Algo Trading Bot V2")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine")
    parser.add_argument("--live", action="store_true", help="Alias for ALPACA_LIVE=true")
    args = parser.parse_args()

    if args.live:
        import os
        os.environ["ALPACA_LIVE"] = "true"

    # Backtest mode
    if args.backtest:
        from backtester import run_backtest
        database.init_db()
        run_backtest()
        return

    console.print("[bold cyan]Starting Algo Trading Bot V2...[/bold cyan]\n")

    # Initialize database
    database.init_db()
    database.migrate_from_json()

    # Startup checks
    info = startup_checks()

    # Initialize components
    regime_detector = MarketRegime()
    orb = ORBStrategy()
    vwap = VWAPStrategy()
    momentum = MomentumStrategy()
    risk = RiskManager()

    # Initialize risk with account info
    risk.reset_daily(info["equity"], info["cash"])

    # Load persisted state from DB
    risk.load_from_db()

    # Load filters
    console.print("Loading earnings calendar...")
    try:
        load_earnings_cache(config.SYMBOLS)
    except Exception as e:
        console.print(f"[yellow]Earnings filter load failed: {e} (continuing without)[/yellow]")

    console.print("Loading correlation data...")
    try:
        load_correlation_cache(config.SYMBOLS)
    except Exception as e:
        console.print(f"[yellow]Correlation filter load failed: {e} (continuing without)[/yellow]")

    start_time = now_et()
    last_scan = None
    last_state_save = now_et()
    last_analytics_update = now_et()
    last_day = now_et().date()
    eod_summary_printed = False
    current_analytics = None

    console.print(f"\n[bold green]Bot V2 is running. Press Ctrl+C to stop.[/bold green]")
    console.print(f"[dim]Strategies: ORB + VWAP + {'MOMENTUM' if config.ALLOW_MOMENTUM else 'MOMENTUM(disabled)'}[/dim]\n")

    # Stop logging to terminal — dashboard takes over
    logging.getLogger().removeHandler(_stream_handler)

    try:
        with Live(
            build_dashboard(risk, regime_detector.regime, start_time, now_et(), last_scan,
                          len(config.SYMBOLS), current_analytics, get_excluded_count()),
            console=console,
            refresh_per_second=0.2,
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
                    momentum.reset_daily()
                    try:
                        account = get_account()
                        risk.reset_daily(float(account.equity), float(account.cash))
                    except Exception as e:
                        logger.error(f"Failed to reset daily: {e}")
                    last_day = current.date()
                    eod_summary_printed = False

                    # Refresh daily caches
                    try:
                        load_earnings_cache(config.SYMBOLS)
                        load_correlation_cache(config.SYMBOLS)
                    except Exception as e:
                        logger.error(f"Failed to refresh daily caches: {e}")

                # Update regime
                regime = regime_detector.update(current)

                # Market hours logic
                if is_market_hours(current_time):

                    # ORB recording period (9:30-10:00)
                    if is_orb_recording_period(current_time):
                        if not orb.ranges_recorded and current_time >= time(9, 55):
                            orb.record_opening_ranges(config.STANDARD_SYMBOLS, current)

                    # Trading hours (10:00-15:45)
                    if is_trading_hours(current_time):
                        signals = []

                        # Run strategies based on regime
                        if regime in ("BULLISH", "UNKNOWN"):
                            orb_signals = orb.scan(config.STANDARD_SYMBOLS, current)
                            signals.extend(orb_signals)

                        # VWAP runs on all symbols in all regimes
                        vwap_signals = vwap.scan(config.SYMBOLS, current, regime)
                        signals.extend(vwap_signals)

                        # Momentum: once daily at 10:30 AM
                        if (config.ALLOW_MOMENTUM
                            and current_time >= config.MOMENTUM_SCAN_TIME
                            and not momentum.scanned_today):
                            mom_signals = momentum.scan(
                                config.STANDARD_SYMBOLS, current, regime_detector
                            )
                            signals.extend(mom_signals)

                        # Process signals
                        if signals:
                            process_signals(signals, risk, regime, current)

                        # Check VWAP time stops
                        expired = check_vwap_time_stops(risk.open_trades, current)
                        for symbol in expired:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="time_stop")

                        # Check momentum max hold
                        expired_mom = check_momentum_max_hold(risk.open_trades, current)
                        for symbol in expired_mom:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="max_hold")

                    # ORB exit time (15:45)
                    if current_time >= config.ORB_EXIT_TIME:
                        closed = close_orb_positions(risk.open_trades, current)
                        for symbol in closed:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="eod_close")

                    # Sync with broker
                    sync_positions_with_broker(risk, current)

                    # Check circuit breaker
                    risk.check_circuit_breaker()

                    last_scan = current

                # EOD summary + daily snapshot at 16:15
                if current_time >= config.EOD_SUMMARY_TIME and not eod_summary_printed:
                    summary = risk.get_day_summary()
                    print_day_summary(summary)
                    logger.info(f"Day summary: {summary}")

                    # Save daily snapshot to DB
                    try:
                        wr = summary.get("win_rate", 0) if summary.get("trades", 0) > 0 else 0
                        database.save_daily_snapshot(
                            date=current.strftime("%Y-%m-%d"),
                            portfolio_value=risk.current_equity,
                            cash=risk.current_cash,
                            day_pnl=risk.day_pnl * risk.starting_equity,
                            day_pnl_pct=risk.day_pnl,
                            total_trades=summary.get("trades", 0),
                            win_rate=wr,
                            sharpe_rolling=current_analytics.get("sharpe_7d", 0) if current_analytics else 0,
                        )
                    except Exception as e:
                        logger.error(f"Failed to save daily snapshot: {e}")

                    eod_summary_printed = True

                # Save state to DB periodically
                if (current - last_state_save).total_seconds() >= config.STATE_SAVE_INTERVAL_SEC:
                    try:
                        database.save_open_positions(risk.open_trades)
                    except Exception as e:
                        logger.error(f"Failed to save state: {e}")
                    last_state_save = current

                # Update analytics every 5 minutes
                if (current - last_analytics_update).total_seconds() >= 300:
                    try:
                        current_analytics = analytics_mod.compute_analytics()
                    except Exception as e:
                        logger.error(f"Failed to compute analytics: {e}")
                    last_analytics_update = current

                # Update dashboard
                live.update(
                    build_dashboard(
                        risk, regime, start_time, current, last_scan,
                        len(config.SYMBOLS), current_analytics, get_excluded_count(),
                    )
                )

                # Sleep until next scan
                time_mod.sleep(config.SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print("[green]State saved. Bot stopped.[/green]")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print(f"[bold red]Bot crashed: {e}[/bold red]")
        console.print("[yellow]State saved. Restart with: python main.py[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
