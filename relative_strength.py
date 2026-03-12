"""V3: Relative Strength Ranking — filter signals by sector-relative performance."""

import logging
from datetime import datetime

import config
from data import get_snapshot

logger = logging.getLogger(__name__)

# Cache: symbol -> today's return %
_day_returns: dict[str, float] = {}
_cache_time: datetime | None = None
_CACHE_TTL_SEC = 120  # Refresh every 2 minutes


class RelativeStrengthTracker:
    """Track intraday relative strength vs sector ETF and SPY."""

    def __init__(self):
        self.spy_return: float = 0.0
        self._sector_returns: dict[str, float] = {}

    def update(self, now: datetime):
        """Refresh sector ETF and SPY returns. Called every scan cycle."""
        global _day_returns, _cache_time

        if _cache_time and (now - _cache_time).total_seconds() < _CACHE_TTL_SEC:
            return

        try:
            # Get SPY return
            self.spy_return = self._get_day_return("SPY")

            # Get sector ETF returns
            sector_etfs = set(config.SECTOR_MAP.values())
            for etf in sector_etfs:
                self._sector_returns[etf] = self._get_day_return(etf)

            _cache_time = now
        except Exception as e:
            logger.error(f"RS update failed: {e}")

    def score(self, symbol: str) -> float:
        """Calculate relative strength score from -1 to +1.

        +1 = strongly outperforming sector and SPY
        -1 = strongly underperforming
        """
        try:
            symbol_return = self._get_day_return(symbol)
            sector_etf = config.SECTOR_MAP.get(symbol, "SPY")
            sector_return = self._sector_returns.get(sector_etf, self.spy_return)

            vs_sector = symbol_return - sector_return
            vs_spy = symbol_return - self.spy_return

            # Weighted score: 60% vs sector, 40% vs SPY
            # Normalize: 2% diff = max score
            raw_score = (vs_sector * 0.6 + vs_spy * 0.4) / 0.02
            return max(-1.0, min(1.0, raw_score))

        except Exception:
            return 0.0  # Neutral on error

    def top_leaders(self, symbols: list[str], n: int = 5) -> list[tuple[str, float]]:
        """Get top N symbols by relative strength."""
        scored = []
        for sym in symbols:
            if sym in config.LEVERAGED_ETFS:
                continue
            s = self.score(sym)
            scored.append((sym, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def _get_day_return(self, symbol: str) -> float:
        """Get today's return for a symbol using cached snapshots."""
        global _day_returns

        if symbol in _day_returns:
            return _day_returns[symbol]

        try:
            snap = get_snapshot(symbol)
            if snap is None:
                return 0.0

            # Get current price and previous close
            current = float(snap.latest_trade.price) if snap.latest_trade else 0
            prev_close = float(snap.previous_daily_bar.close) if snap.previous_daily_bar else current

            if prev_close <= 0:
                return 0.0

            ret = (current - prev_close) / prev_close
            _day_returns[symbol] = ret
            return ret

        except Exception:
            return 0.0

    @staticmethod
    def clear_cache():
        """Clear daily return cache. Call at start of each day."""
        global _day_returns, _cache_time
        _day_returns.clear()
        _cache_time = None
