"""Entry point V3 — ML filter, dynamic allocation, WebSocket, shorts, Gap & Go, RS filter."""

import argparse
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
from strategies import MarketRegime, ORBStrategy, VWAPStrategy, MomentumStrategy, GapGoStrategy, Signal
from risk import RiskManager, TradeRecord
from execution import (
    submit_bracket_order,
    close_position,
    close_orb_positions,
    check_vwap_time_stops,
    check_momentum_max_hold,
    cancel_all_open_orders,
    can_short,
    close_gap_go_positions,
)
from dashboard import (
    build_dashboard,
    print_day_summary,
    print_startup_info,
    console,
)
from earnings import load_earnings_cache, has_earnings_soon, get_excluded_count
from correlation import load_correlation_cache, is_too_correlated

# V3 imports (conditional)
try:
    from ml_filter import ml_filter, extract_live_features
except ImportError:
    ml_filter = None

try:
    from relative_strength import RelativeStrengthTracker
except ImportError:
    RelativeStrengthTracker = None

try:
    from position_monitor import PositionMonitor
except ImportError:
    PositionMonitor = None

try:
    import notifications
except ImportError:
    notifications = None

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
    rs_tracker=None,
    ws_monitor=None,
    market_data: dict | None = None,
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

        # V3: ML signal filter
        if config.USE_ML_FILTER and ml_filter and ml_filter._active.get(signal.strategy):
            try:
                features = extract_live_features(signal, regime, market_data or {})
                prob = ml_filter.should_trade(signal.strategy, features)
                if prob < config.ML_PROBABILITY_THRESHOLD:
                    skip_reason = f"ml_filter_{prob:.2f}"
                    logger.info(f"ML filter rejected {signal.symbol} ({signal.strategy}): prob={prob:.2f}")
                    database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                    continue
            except Exception as e:
                logger.warning(f"ML filter error for {signal.symbol}: {e}")

        # V3: Relative strength filter
        if config.USE_RS_FILTER and rs_tracker:
            try:
                rs_score = rs_tracker.score(signal.symbol)
                if signal.side == "buy" and rs_score < config.RS_LONG_THRESHOLD:
                    skip_reason = f"rs_weak_{rs_score:.2f}"
                    database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                    continue
                elif signal.side == "sell" and rs_score > config.RS_SHORT_THRESHOLD:
                    skip_reason = f"rs_strong_{rs_score:.2f}"
                    database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                    continue
            except Exception as e:
                logger.warning(f"RS filter error for {signal.symbol}: {e}")

        # V3: Short selling pre-check
        if signal.side == "sell":
            if not config.ALLOW_SHORT:
                skip_reason = "shorting_disabled"
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                continue
            shortable, short_reason = can_short(signal.symbol, 1, signal.entry_price)
            if not shortable:
                skip_reason = f"short_blocked_{short_reason}"
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                continue

        # Check risk limits
        allowed, reason = risk.can_open_trade(strategy=signal.strategy)
        if not allowed:
            skip_reason = reason
            logger.info(f"Trade blocked for {signal.symbol}: {reason}")
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            continue

        # Calculate position size (V3: strategy-weighted + short multiplier)
        qty = risk.calculate_position_size(
            signal.entry_price, signal.stop_loss, regime,
            strategy=signal.strategy, side=signal.side,
        )
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
        elif signal.strategy == "GAP_GO":
            time_stop = datetime.combine(now.date(), config.GAP_EXIT_TIME, tzinfo=config.ET)

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

        # V3: Subscribe to WebSocket monitoring
        if ws_monitor:
            ws_monitor.subscribe(signal.symbol)

        # V3: Telegram notification
        if notifications and config.TELEGRAM_ENABLED:
            try:
                notifications.notify_trade_opened(trade)
            except Exception as e:
                logger.warning(f"Telegram notification failed: {e}")

        # Log signal as acted on
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, True, "")


def sync_positions_with_broker(risk: RiskManager, now: datetime, ws_monitor=None):
    """Sync open trades with actual broker positions to detect fills/stops."""
    try:
        broker_positions = {p.symbol: p for p in get_positions()}
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return

    for symbol in list(risk.open_trades.keys()):
        if symbol not in broker_positions:
            trade = risk.open_trades[symbol]
            risk.close_trade(symbol, trade.entry_price, now, exit_reason="broker_sync")
            logger.info(f"Position {symbol} no longer at broker — marking closed")

            # V3: Unsubscribe from WS and notify
            if ws_monitor:
                ws_monitor.unsubscribe(symbol)
            if notifications and config.TELEGRAM_ENABLED:
                try:
                    notifications.notify_trade_closed(trade)
                except Exception:
                    pass

    try:
        account = get_account()
        risk.update_equity(float(account.equity), float(account.cash))
    except Exception as e:
        logger.error(f"Failed to update account: {e}")


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Algo Trading Bot V3")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward test")
    parser.add_argument("--train-ml", action="store_true", help="Train ML signal filter")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
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

    # Walk-forward mode
    if args.walkforward:
        from backtester import walk_forward_test
        database.init_db()
        walk_forward_test()
        return

    # Train ML mode
    if args.train_ml:
        database.init_db()
        if ml_filter:
            ml_filter.retrain_all()
        else:
            console.print("[red]ML filter module not available[/red]")
        return

    # Optimize mode
    if args.optimize:
        from optimizer import weekly_optimization
        database.init_db()
        weekly_optimization()
        return

    console.print("[bold cyan]Starting Algo Trading Bot V3...[/bold cyan]\n")

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
    gap_go = GapGoStrategy() if config.GAP_GO_ENABLED else None
    risk = RiskManager()

    # Initialize risk with account info
    risk.reset_daily(info["equity"], info["cash"])

    # Load persisted state from DB
    risk.load_from_db()

    # V3: Initialize relative strength tracker
    rs_tracker = None
    if config.USE_RS_FILTER and RelativeStrengthTracker:
        rs_tracker = RelativeStrengthTracker()
        console.print("[green]Relative strength tracker initialized.[/green]")

    # V3: Load ML models
    if config.USE_ML_FILTER and ml_filter:
        try:
            ml_filter.load_models()
            active = [s for s, a in ml_filter._active.items() if a]
            if active:
                console.print(f"[green]ML filter loaded for: {', '.join(active)}[/green]")
            else:
                console.print("[yellow]ML filter: no trained models yet (need 100+ trades)[/yellow]")
        except Exception as e:
            console.print(f"[yellow]ML filter load failed: {e}[/yellow]")

    # V3: Calculate initial strategy weights
    if config.DYNAMIC_ALLOCATION:
        try:
            risk.update_strategy_weights()
            weights = risk.get_strategy_weights()
            if weights:
                weight_str = ", ".join(f"{s}: {w:.0%}" for s, w in weights.items())
                console.print(f"[green]Capital allocation: {weight_str}[/green]")
        except Exception as e:
            console.print(f"[yellow]Dynamic allocation init failed: {e}[/yellow]")

    # V3: Start WebSocket position monitor
    ws_monitor = None
    if config.WEBSOCKET_MONITORING and PositionMonitor:
        ws_monitor = PositionMonitor(risk)
        ws_monitor.set_close_callback(
            lambda symbol, reason: _handle_ws_close(symbol, reason, risk, ws_monitor)
        )
        # Subscribe to existing open positions
        for symbol in risk.open_trades:
            ws_monitor.subscribe(symbol)
        ws_monitor.start()
        console.print("[green]WebSocket position monitor started.[/green]")

    # V3: Start web dashboard
    if config.WEB_DASHBOARD_ENABLED:
        try:
            from web_dashboard import start_web_dashboard
            start_web_dashboard()
            console.print(f"[green]Web dashboard: http://localhost:{config.WEB_DASHBOARD_PORT}[/green]")
        except Exception as e:
            console.print(f"[yellow]Web dashboard failed to start: {e}[/yellow]")

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
    gap_candidates_found = False
    gap_first_candle_recorded = False
    allocation_updated_today = False
    last_sunday_task = None  # Track Sunday midnight ML/optimize runs

    # V3 feature flags summary
    v3_features = []
    if config.USE_ML_FILTER:
        v3_features.append("ML Filter")
    if config.DYNAMIC_ALLOCATION:
        v3_features.append("Dynamic Alloc")
    if config.WEBSOCKET_MONITORING:
        v3_features.append("WebSocket")
    if config.ALLOW_SHORT:
        v3_features.append("Shorts")
    if config.GAP_GO_ENABLED:
        v3_features.append("Gap&Go")
    if config.USE_RS_FILTER:
        v3_features.append("RS Filter")
    if config.TELEGRAM_ENABLED:
        v3_features.append("Telegram")
    if config.WEB_DASHBOARD_ENABLED:
        v3_features.append("Web Dash")
    if config.AUTO_OPTIMIZE:
        v3_features.append("Optimizer")

    features_str = ", ".join(v3_features) if v3_features else "none"
    console.print(f"\n[bold green]Bot V3 is running. Press Ctrl+C to stop.[/bold green]")
    console.print(f"[dim]Strategies: ORB + VWAP + {'MOMENTUM' if config.ALLOW_MOMENTUM else 'MOMENTUM(off)'} + {'GAP_GO' if config.GAP_GO_ENABLED else 'GAP_GO(off)'}[/dim]")
    console.print(f"[dim]V3 features: {features_str}[/dim]\n")

    # Stop logging to terminal — dashboard takes over
    logging.getLogger().removeHandler(_stream_handler)

    try:
        with Live(
            build_dashboard(risk, regime_detector.regime, start_time, now_et(), last_scan,
                          len(config.SYMBOLS), current_analytics, get_excluded_count(),
                          risk.get_strategy_weights()),
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
                    if gap_go:
                        gap_go.reset_daily()
                    gap_candidates_found = False
                    gap_first_candle_recorded = False
                    allocation_updated_today = False

                    if rs_tracker:
                        rs_tracker.clear_cache()

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

                # V3: Sunday midnight tasks (ML retrain + parameter optimization)
                if (current.weekday() == 6 and current_time >= time(0, 0)
                        and last_sunday_task != current.date()):
                    last_sunday_task = current.date()

                    if config.USE_ML_FILTER and ml_filter:
                        try:
                            logger.info("Sunday: retraining ML models...")
                            ml_filter.retrain_all()
                            if notifications and config.TELEGRAM_ENABLED:
                                notifications.notify_ml_retrain(ml_filter._active)
                        except Exception as e:
                            logger.error(f"ML retrain failed: {e}")

                    if config.AUTO_OPTIMIZE:
                        try:
                            logger.info("Sunday: running parameter optimization...")
                            from optimizer import weekly_optimization
                            weekly_optimization()
                        except Exception as e:
                            logger.error(f"Parameter optimization failed: {e}")

                # Update regime
                regime = regime_detector.update(current)

                # V3: Update RS tracker periodically
                if rs_tracker and is_market_hours(current_time):
                    try:
                        rs_tracker.update(current)
                    except Exception as e:
                        logger.warning(f"RS tracker update failed: {e}")

                # V3: Daily capital allocation update at 9:00 AM
                if (config.DYNAMIC_ALLOCATION
                        and not allocation_updated_today
                        and current_time >= config.ALLOCATION_RECALC_TIME):
                    try:
                        risk.update_strategy_weights()
                        allocation_updated_today = True
                    except Exception as e:
                        logger.error(f"Failed to update strategy weights: {e}")

                # Market hours logic
                if is_market_hours(current_time):

                    # V3: Gap & Go pre-market scan at 9:00 AM
                    if (gap_go and not gap_candidates_found
                            and current_time >= config.GAP_PREMARKET_SCAN_TIME):
                        try:
                            gap_go.find_gap_candidates(config.SYMBOLS, current)
                            gap_candidates_found = True
                            logger.info(f"Gap & Go: found {len(gap_go.candidates)} candidates")
                        except Exception as e:
                            logger.error(f"Gap & Go pre-market scan failed: {e}")

                    # ORB recording period (9:30-10:00)
                    if is_orb_recording_period(current_time):
                        if not orb.ranges_recorded and current_time >= time(9, 55):
                            orb.record_opening_ranges(config.STANDARD_SYMBOLS, current)

                        # V3: Gap & Go first candle recording at 9:45
                        if (gap_go and not gap_first_candle_recorded
                                and current_time >= time(9, 45)):
                            try:
                                gap_go.record_first_candle(current)
                                gap_first_candle_recorded = True
                            except Exception as e:
                                logger.error(f"Gap & Go first candle record failed: {e}")

                    # Trading hours (10:00-15:45)
                    if is_trading_hours(current_time):
                        signals = []

                        # Run strategies based on regime
                        if regime in ("BULLISH", "UNKNOWN"):
                            orb_signals = orb.scan(config.STANDARD_SYMBOLS, current)
                            signals.extend(orb_signals)

                        # V3: ORB short signals in bearish regime
                        if config.ALLOW_SHORT and regime in ("BEARISH", "UNKNOWN"):
                            orb_short_signals = orb.scan(config.STANDARD_SYMBOLS, current, regime=regime)
                            # Only add short signals not already in signals list
                            for sig in orb_short_signals:
                                if sig.side == "sell" and sig.symbol not in [s.symbol for s in signals]:
                                    signals.append(sig)

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

                        # V3: Gap & Go scan (9:45-11:30)
                        if (gap_go and gap_first_candle_recorded
                                and current_time >= config.GAP_ENTRY_TIME
                                and current_time < config.GAP_EXIT_TIME):
                            try:
                                gap_signals = gap_go.scan(current)
                                signals.extend(gap_signals)
                            except Exception as e:
                                logger.error(f"Gap & Go scan failed: {e}")

                        # Process signals
                        if signals:
                            process_signals(
                                signals, risk, regime, current,
                                rs_tracker=rs_tracker,
                                ws_monitor=ws_monitor,
                            )

                        # Check VWAP time stops
                        expired = check_vwap_time_stops(risk.open_trades, current)
                        for symbol in expired:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="time_stop")
                                if ws_monitor:
                                    ws_monitor.unsubscribe(symbol)

                        # Check momentum max hold
                        expired_mom = check_momentum_max_hold(risk.open_trades, current)
                        for symbol in expired_mom:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="max_hold")
                                if ws_monitor:
                                    ws_monitor.unsubscribe(symbol)

                        # V3: Gap & Go time stop at 11:30
                        if gap_go and current_time >= config.GAP_EXIT_TIME:
                            closed_gaps = close_gap_go_positions(risk.open_trades, current)
                            for symbol in closed_gaps:
                                if symbol in risk.open_trades:
                                    risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="gap_time_stop")
                                    if ws_monitor:
                                        ws_monitor.unsubscribe(symbol)

                    # ORB exit time (15:45)
                    if current_time >= config.ORB_EXIT_TIME:
                        closed = close_orb_positions(risk.open_trades, current)
                        for symbol in closed:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="eod_close")
                                if ws_monitor:
                                    ws_monitor.unsubscribe(symbol)

                    # Sync with broker (use polling if WS not connected)
                    if not (ws_monitor and ws_monitor.is_connected):
                        sync_positions_with_broker(risk, current, ws_monitor)
                    else:
                        # Still update equity from account
                        try:
                            account = get_account()
                            risk.update_equity(float(account.equity), float(account.cash))
                        except Exception as e:
                            logger.error(f"Failed to update account: {e}")

                    # Check circuit breaker
                    if risk.check_circuit_breaker():
                        if notifications and config.TELEGRAM_ENABLED:
                            try:
                                notifications.notify_circuit_breaker(risk.day_pnl)
                            except Exception:
                                pass

                    last_scan = current

                # EOD summary + daily snapshot at 16:15
                if current_time >= config.EOD_SUMMARY_TIME and not eod_summary_printed:
                    summary = risk.get_day_summary()
                    print_day_summary(summary)
                    logger.info(f"Day summary: {summary}")

                    # V3: Telegram daily summary
                    if notifications and config.TELEGRAM_ENABLED:
                        try:
                            notifications.notify_daily_summary(summary, risk.current_equity)
                        except Exception as e:
                            logger.warning(f"Telegram daily summary failed: {e}")

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

                        # V3: Drawdown warning
                        if current_analytics and notifications and config.TELEGRAM_ENABLED:
                            dd = current_analytics.get("max_drawdown", 0)
                            if dd > 0.05:
                                try:
                                    notifications.notify_drawdown_warning(dd)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error(f"Failed to compute analytics: {e}")
                    last_analytics_update = current

                # Update dashboard
                live.update(
                    build_dashboard(
                        risk, regime, start_time, current, last_scan,
                        len(config.SYMBOLS), current_analytics, get_excluded_count(),
                        risk.get_strategy_weights(),
                    )
                )

                # Sleep until next scan
                time_mod.sleep(config.SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        if ws_monitor:
            ws_monitor.stop()
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print("[green]State saved. Bot stopped.[/green]")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        if ws_monitor:
            ws_monitor.stop()
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print(f"[bold red]Bot crashed: {e}[/bold red]")
        console.print("[yellow]State saved. Restart with: python main.py[/yellow]")
        sys.exit(1)


def _handle_ws_close(symbol: str, reason: str, risk: RiskManager, ws_monitor):
    """Callback for WebSocket-triggered position closes."""
    if symbol in risk.open_trades:
        trade = risk.open_trades[symbol]
        try:
            close_position(symbol)
        except Exception as e:
            logger.error(f"WS close failed for {symbol}: {e}")
            return
        risk.close_trade(symbol, trade.entry_price, now_et(), exit_reason=reason)
        ws_monitor.unsubscribe(symbol)

        if notifications and config.TELEGRAM_ENABLED:
            try:
                notifications.notify_trade_closed(trade)
            except Exception:
                pass


if __name__ == "__main__":
    main()
