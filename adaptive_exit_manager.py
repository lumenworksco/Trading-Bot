"""Adaptive exit manager — VIX-aware exit parameters and exit decisions.

Adjusts z-score thresholds, time multipliers, and partial-exit triggers
based on the current VIX regime.
"""

import logging
from datetime import datetime

import config
from analytics.ou_tools import compute_zscore

logger = logging.getLogger(__name__)


class AdaptiveExitManager:
    """VIX-aware exit management for mean-reversion positions."""

    # ------------------------------------------------------------------ #
    #  VIX-regime exit parameters
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_exit_params(vix: float, half_life_hours: float) -> dict:
        """Return exit thresholds based on the current VIX level.

        Args:
            vix: Current VIX index value.
            half_life_hours: Estimated OU half-life in hours.

        Returns:
            dict with z_exit, time_mult, partial_z.
        """
        if vix < 15:
            # Calm market — let trades run, tight reversion target
            return {'z_exit': 0.3, 'time_mult': 3.0, 'partial_z': 0.8}
        elif vix <= 25:
            # Normal
            return {'z_exit': 0.2, 'time_mult': 2.0, 'partial_z': 0.5}
        elif vix <= 35:
            # Elevated — take profits faster
            return {'z_exit': 0.5, 'time_mult': 1.0, 'partial_z': 1.0}
        else:
            # Crisis — aggressive exits
            return {'z_exit': 0.8, 'time_mult': 0.5, 'partial_z': 1.5}

    # ------------------------------------------------------------------ #
    #  Exit decision
    # ------------------------------------------------------------------ #

    def should_exit(
        self,
        position_side: str,
        position_entry_time: datetime,
        current_price: float,
        ou_params: dict,
        vix: float,
        partial_exits: int,
    ) -> tuple[str, str]:
        """Decide whether to exit a mean-reversion position.

        Args:
            position_side: 'buy' (long) or 'sell' (short).
            position_entry_time: When the position was opened.
            current_price: Latest market price.
            ou_params: dict with keys mu, sigma, half_life (hours).
            vix: Current VIX level.
            partial_exits: Number of partial exits already taken.

        Returns:
            (exit_reason, exit_type) where exit_type is
            'full', 'partial', or 'hold'.
        """
        mu = ou_params.get('mu', current_price)
        sigma = ou_params.get('sigma', 0.0)
        half_life_hours = ou_params.get('half_life', 24.0)

        z = compute_zscore(current_price, mu, sigma)
        params = self.get_exit_params(vix, half_life_hours)

        # --- Fixed stop: never adaptive ---
        if abs(z) > config.MR_ZSCORE_STOP:
            logger.info("Adaptive exit: z=%.2f exceeds stop %.1f",
                        z, config.MR_ZSCORE_STOP)
            return ('stop_loss', 'full')

        # --- Full reversion ---
        # For a long position z < 0 at entry; full reversion = z near 0
        # For a short position z > 0 at entry; full reversion = z near 0
        if abs(z) <= params['z_exit']:
            logger.info("Adaptive exit: full reversion z=%.2f <= %.2f",
                        z, params['z_exit'])
            return ('full_reversion', 'full')

        # --- Partial exit ---
        if partial_exits == 0 and abs(z) <= params['partial_z']:
            logger.info("Adaptive exit: partial at z=%.2f <= %.2f",
                        z, params['partial_z'])
            return ('partial_reversion', 'partial')

        # --- Time stop ---
        now = datetime.now(position_entry_time.tzinfo)
        hold_hours = (now - position_entry_time).total_seconds() / 3600.0
        max_hours = half_life_hours * params['time_mult']
        if hold_hours > max_hours:
            logger.info(
                "Adaptive exit: time stop — held %.1fh > %.1fh",
                hold_hours, max_hours,
            )
            return ('time_stop', 'full')

        return ('hold', 'hold')
