"""Earnings filter — skip trades near earnings announcements."""

import logging
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)

# Daily cache: symbol -> has_earnings_soon (bool)
_earnings_cache: dict[str, bool] = {}
_cache_date: str = ""

# ETFs don't report earnings — skip them to avoid yfinance 404 spam
_KNOWN_ETFS = {
    # Core ETFs
    "SPY", "QQQ", "IWM", "DIA", "ARKK", "GLD", "SLV", "TLT",
    # Sector ETFs
    "XLF", "XLE", "XLK", "XLV", "XLB", "XLI", "XLU", "XLRE", "XLC", "XLP", "XLY",
    "SOXX", "SMH", "IBB",
    # Leveraged / Inverse ETFs
    "SOXL", "TQQQ", "SPXL", "UVXY", "SQQQ", "TNA", "FAS", "FAZ", "LABU", "LABD",
}


def load_earnings_cache(symbols: list[str]):
    """Refresh earnings cache for all symbols. Call once per day."""
    global _earnings_cache, _cache_date

    today = datetime.now(config.ET).strftime("%Y-%m-%d")
    if _cache_date == today and _earnings_cache:
        return  # Already loaded today

    logger.info(f"Loading earnings calendar for {len(symbols)} symbols...")
    _earnings_cache.clear()
    excluded = 0

    # Suppress yfinance's noisy internal logging during batch check
    yf_logger = logging.getLogger("yfinance")
    prev_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)

    try:
        for symbol in symbols:
            # ETFs don't have earnings — skip to avoid 404 errors
            if symbol in _KNOWN_ETFS:
                _earnings_cache[symbol] = False
                continue
            try:
                has_earnings = _check_earnings(symbol, config.EARNINGS_FILTER_DAYS)
                _earnings_cache[symbol] = has_earnings
                if has_earnings:
                    excluded += 1
            except Exception:
                _earnings_cache[symbol] = False  # If we can't determine, allow
    finally:
        yf_logger.setLevel(prev_level)

    _cache_date = today
    logger.info(f"Earnings filter: {excluded} symbols excluded ({len(_KNOWN_ETFS)} ETFs auto-skipped)")


def _check_earnings(symbol: str, days: int = 2) -> bool:
    """Check if symbol has earnings within N days using yfinance."""
    try:
        import yfinance as yf
        import pandas as pd

        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is None or (hasattr(cal, 'empty') and cal.empty):
            return False

        # yfinance calendar format varies by version
        earnings_date = None
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.columns:
                earnings_date = cal["Earnings Date"].iloc[0]
            elif len(cal.columns) > 0:
                # Some versions put date in first column
                earnings_date = cal.iloc[0, 0]
        elif isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date", [None])[0]

        if earnings_date is None:
            return False

        if isinstance(earnings_date, str):
            earnings_date = pd.Timestamp(earnings_date)

        today = datetime.now(config.ET).date()
        if hasattr(earnings_date, 'date'):
            earnings_day = earnings_date.date()
        else:
            earnings_day = earnings_date

        days_until = (earnings_day - today).days
        return 0 <= days_until <= days

    except Exception:
        return False  # If we can't determine, allow the trade


def has_earnings_soon(symbol: str) -> bool:
    """Check if symbol has earnings soon. Uses cache if available."""
    if symbol in _earnings_cache:
        return _earnings_cache[symbol]
    # Not in cache — check live (shouldn't happen if cache is loaded)
    return _check_earnings(symbol, config.EARNINGS_FILTER_DAYS)


def get_excluded_count() -> int:
    """Get count of symbols excluded due to earnings."""
    return sum(1 for v in _earnings_cache.values() if v)
