"""V3: Weekly parameter auto-optimization — grid search over strategy params."""

import logging
from datetime import datetime
from itertools import product

import numpy as np

import config
import database
import notifications

logger = logging.getLogger(__name__)


def weekly_optimization():
    """Grid search over strategy parameters using recent backtest data.

    Called every Sunday at midnight. Only updates params if Sharpe improves > 10%.
    """
    logger.info("Starting weekly parameter optimization...")

    param_grids = {
        "ORB": {
            "ORB_VOLUME_MULTIPLIER": [1.3, 1.5, 1.8],
            "ORB_TAKE_PROFIT_MULT": [1.2, 1.5, 2.0],
            "ORB_STOP_LOSS_MULT": [0.4, 0.5, 0.6],
        },
        "VWAP": {
            "VWAP_BAND_STD": [1.2, 1.5, 2.0],
            "VWAP_RSI_OVERSOLD": [35, 40, 45],
            "VWAP_RSI_OVERBOUGHT": [55, 60, 65],
        },
    }

    results = {}

    for strategy, grid in param_grids.items():
        logger.info(f"Optimizing {strategy}...")

        # Get current performance baseline
        current_sharpe = _get_current_sharpe(strategy)
        old_params = _get_current_params(strategy, grid)

        best_sharpe = -999.0
        best_params = None

        keys = list(grid.keys())
        values = list(grid.values())

        for combo in product(*values):
            param_dict = dict(zip(keys, combo))

            # Simulate with these params
            result = _evaluate_params(strategy, param_dict)
            if result is None:
                continue

            sharpe = result.get("sharpe", 0)
            win_rate = result.get("win_rate", 0)

            # Must have reasonable win rate
            if win_rate < 0.45:
                continue

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = param_dict

        # Check if improvement is significant
        applied = False
        if best_params and current_sharpe > 0:
            improvement = (best_sharpe - current_sharpe) / abs(current_sharpe)
            if improvement > config.OPTIMIZE_MIN_IMPROVEMENT:
                _apply_params(strategy, best_params)
                applied = True
                logger.info(
                    f"Optimized {strategy}: Sharpe {current_sharpe:.2f} -> {best_sharpe:.2f} "
                    f"({improvement:.0%} improvement). Params: {best_params}"
                )
                notifications.notify_optimization(
                    strategy, current_sharpe, best_sharpe, best_params
                )
            else:
                logger.info(
                    f"{strategy}: Best Sharpe {best_sharpe:.2f} vs current {current_sharpe:.2f} "
                    f"({improvement:.0%} < {config.OPTIMIZE_MIN_IMPROVEMENT:.0%} threshold). No change."
                )
        elif best_params:
            # No current baseline — apply if positive
            if best_sharpe > 0.5:
                _apply_params(strategy, best_params)
                applied = True

        # Log to database
        try:
            database.log_optimization(
                strategy=strategy,
                old_params=old_params,
                new_params=best_params or old_params,
                old_sharpe=current_sharpe,
                new_sharpe=best_sharpe,
                applied=applied,
            )
        except Exception as e:
            logger.error(f"Failed to log optimization: {e}")

        results[strategy] = {
            "old_sharpe": current_sharpe,
            "new_sharpe": best_sharpe,
            "applied": applied,
            "params": best_params,
        }

    logger.info(f"Optimization complete: {results}")
    return results


def _get_current_sharpe(strategy: str) -> float:
    """Get current Sharpe ratio from recent trades."""
    trades = database.get_recent_trades_by_strategy(
        strategy, days=config.OPTIMIZE_LOOKBACK_WEEKS * 7
    )
    if len(trades) < 10:
        return 0.0

    from collections import defaultdict
    daily_pnl = defaultdict(float)
    for t in trades:
        date = (t.get("exit_time") or "")[:10]
        if date:
            daily_pnl[date] += t.get("pnl_pct", 0)

    returns = list(daily_pnl.values())
    if len(returns) < 5:
        return 0.0

    arr = np.array(returns)
    rf = config.BACKTEST_RISK_FREE_RATE / 252
    excess = arr - rf
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252))


def _get_current_params(strategy: str, grid: dict) -> dict:
    """Get current config values for the grid parameters."""
    params = {}
    for key in grid:
        # Check runtime params first, then config defaults
        runtime_val = config.get_param(key)
        if runtime_val is not None:
            params[key] = runtime_val
        else:
            params[key] = getattr(config, key, None)
    return params


def _evaluate_params(strategy: str, params: dict) -> dict | None:
    """Evaluate a parameter combination using recent trade data.

    Uses the backtester with specified params on recent data.
    Returns dict with sharpe and win_rate, or None on failure.
    """
    try:
        from backtester import download_data, simulate_orb, simulate_vwap

        # Temporarily apply params
        original = {}
        for key, value in params.items():
            original[key] = getattr(config, key, None)
            setattr(config, key, value)

        # Run simulation on top symbols with 2 months of data
        symbols = config.CORE_SYMBOLS[:10]  # Smaller set for speed

        # Use cached data if available, otherwise quick download
        import yfinance as yf
        from datetime import timedelta

        data = {}
        end = datetime.now()
        start = end - timedelta(weeks=config.OPTIMIZE_LOOKBACK_WEEKS)

        # Suppress yfinance logging
        yf_logger = logging.getLogger("yfinance")
        prev_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)

        try:
            for sym in symbols:
                try:
                    df = yf.Ticker(sym).history(start=start, end=end, interval="1h", auto_adjust=True)
                    if len(df) >= 50:
                        df.columns = [c.lower() for c in df.columns]
                        data[sym] = df
                except Exception:
                    pass
        finally:
            yf_logger.setLevel(prev_level)

        if len(data) < 3:
            return None

        # Run appropriate simulation
        if strategy == "ORB":
            result = simulate_orb(data)
        elif strategy == "VWAP":
            result = simulate_vwap(data)
        else:
            return None

        # Restore original params
        for key, value in original.items():
            setattr(config, key, value)

        return {
            "sharpe": result.sharpe_ratio,
            "win_rate": result.win_rate,
            "trades": result.total_trades,
        }

    except Exception as e:
        # Restore params on error
        for key, value in original.items():
            setattr(config, key, value)
        logger.error(f"Param evaluation failed for {strategy}: {e}")
        return None


def _apply_params(strategy: str, params: dict):
    """Apply optimized parameters to runtime config."""
    for key, value in params.items():
        config.set_param(key, value)
        setattr(config, key, value)
    logger.info(f"Applied optimized params for {strategy}: {params}")
