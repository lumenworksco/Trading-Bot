"""Beta Neutralizer — keep portfolio market-neutral.

Computes weighted beta of all open positions vs SPY.
If portfolio beta drifts beyond ±0.3, generates a hedge signal
(long or short SPY) to neutralize.

Skips first 15 minutes of trading (opening imbalances create transient beta).
"""

import logging
from datetime import datetime, time, timedelta

import config
from strategies.base import Signal

logger = logging.getLogger(__name__)


# Pre-computed approximate betas for common symbols (updated periodically)
# These are rough estimates — in production, compute from recent daily returns
DEFAULT_BETAS = {
    'SPY': 1.0, 'QQQ': 1.15, 'IWM': 1.2, 'DIA': 0.95,
    'AAPL': 1.1, 'MSFT': 1.05, 'GOOGL': 1.15, 'META': 1.3, 'AMZN': 1.2,
    'NVDA': 1.6, 'AMD': 1.5, 'INTC': 1.1, 'TSLA': 1.8,
    'JPM': 1.1, 'BAC': 1.2, 'GS': 1.3, 'V': 0.9, 'MA': 0.9,
    'NFLX': 1.2, 'PYPL': 1.3, 'SQ': 1.5, 'COIN': 2.0,
    'XOM': 0.8, 'CVX': 0.8, 'UNH': 0.7, 'LLY': 0.6, 'PFE': 0.7,
    'GLD': 0.05, 'SLV': 0.1, 'TLT': -0.3,
    'SOFI': 1.6, 'PLTR': 1.5, 'CRWD': 1.3, 'PANW': 1.2,
    'SNOW': 1.4, 'DDOG': 1.3, 'NET': 1.4, 'CRM': 1.1, 'NOW': 1.1,
    'UBER': 1.3, 'LYFT': 1.5, 'ABNB': 1.4, 'DASH': 1.3,
    'HOOD': 1.8, 'AFRM': 1.7, 'SMCI': 2.0,
}


class BetaNeutralizer:
    """Monitors and neutralizes portfolio beta."""

    def __init__(self):
        self._last_check = None
        self._portfolio_beta = 0.0
        self._betas: dict[str, float] = dict(DEFAULT_BETAS)

    def get_beta(self, symbol: str) -> float:
        """Get beta for a symbol (from cache or default)."""
        return self._betas.get(symbol, 1.0)

    def compute_portfolio_beta(
        self,
        open_trades: dict,
        prices: dict[str, float],
    ) -> float:
        """
        Compute dollar-weighted beta of all open positions.

        Short positions contribute negative beta.

        Args:
            open_trades: dict of symbol -> TradeRecord
            prices: dict of symbol -> current price

        Returns:
            Portfolio beta (positive = net long market, negative = net short)
        """
        if not open_trades:
            self._portfolio_beta = 0.0
            return 0.0

        total_value = 0.0
        weighted_beta = 0.0

        for symbol, trade in open_trades.items():
            price = prices.get(symbol, trade.entry_price)
            pos_value = abs(trade.qty * price)
            beta = self.get_beta(symbol)

            if trade.side == "sell":
                beta = -beta  # Short contributes negative beta

            weighted_beta += beta * pos_value
            total_value += pos_value

        if total_value < 1e-6:
            self._portfolio_beta = 0.0
            return 0.0

        self._portfolio_beta = weighted_beta / total_value
        return self._portfolio_beta

    def should_skip(self, now: datetime) -> bool:
        """Skip beta check during first 15 minutes of trading."""
        market_open = now.replace(
            hour=config.MARKET_OPEN.hour,
            minute=config.MARKET_OPEN.minute,
            second=0,
        )
        skip_until = market_open + timedelta(minutes=config.BETA_SKIP_FIRST_MINUTES)
        return now < skip_until

    def needs_hedge(self) -> bool:
        """Whether portfolio beta exceeds the neutral band."""
        return abs(self._portfolio_beta) > config.BETA_MAX_ABS

    def compute_hedge_signal(
        self,
        equity: float,
        spy_price: float,
    ) -> Signal | None:
        """
        Generate a SPY hedge signal to neutralize portfolio beta.

        Args:
            equity: Current portfolio value
            spy_price: Current SPY price

        Returns:
            Signal to buy/short SPY, or None if no hedge needed
        """
        if not self.needs_hedge() or spy_price <= 0:
            return None

        beta = self._portfolio_beta

        # Calculate hedge size in dollars
        hedge_dollars = abs(beta) * equity * 0.5  # Partial neutralization
        hedge_qty = int(hedge_dollars / spy_price)

        if hedge_qty <= 0:
            return None

        if beta > config.BETA_MAX_ABS:
            # Portfolio is net long — short SPY to neutralize
            if not config.ALLOW_SHORT:
                logger.info("Beta hedge requires shorting SPY but ALLOW_SHORT is False")
                return None
            return Signal(
                symbol="SPY",
                strategy="BETA_HEDGE",
                side="sell",
                entry_price=spy_price,
                take_profit=spy_price * 0.99,  # Hedge doesn't have traditional TP
                stop_loss=spy_price * 1.03,     # Wide stop — hedge is temporary
                reason=f"beta_hedge: portfolio_beta={beta:.2f}",
                hold_type="day",
            )
        elif beta < -config.BETA_MAX_ABS:
            # Portfolio is net short — buy SPY to neutralize
            return Signal(
                symbol="SPY",
                strategy="BETA_HEDGE",
                side="buy",
                entry_price=spy_price,
                take_profit=spy_price * 1.01,
                stop_loss=spy_price * 0.97,
                reason=f"beta_hedge: portfolio_beta={beta:.2f}",
                hold_type="day",
            )

        return None

    def should_check_now(self, now: datetime) -> bool:
        """Whether it's time for a beta check (every BETA_CHECK_INTERVAL_MIN)."""
        if self._last_check is None:
            self._last_check = now
            return True

        elapsed = (now - self._last_check).total_seconds() / 60
        if elapsed >= config.BETA_CHECK_INTERVAL_MIN:
            self._last_check = now
            return True
        return False

    @property
    def portfolio_beta(self) -> float:
        return self._portfolio_beta

    def reset_daily(self):
        """Reset for new trading day."""
        self._last_check = None
        self._portfolio_beta = 0.0
