"""Consistency Score — single number (0-100) measuring return consistency."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_consistency_score(
    daily_pnl_pcts: list[float],
    sharpe: float = 0.0,
    max_drawdown: float = 0.0,
) -> float:
    """
    Compute consistency score (0-100).

    100 = perfect (positive every day, high Sharpe, no drawdown)
    0 = terrible (random/negative)

    Components (weighted):
    - 40%: Percentage of profitable days
    - 30%: Sharpe ratio normalized (2.0 = perfect)
    - 30%: Max drawdown normalized (-5% = 0)

    Args:
        daily_pnl_pcts: List of daily P&L percentages
        sharpe: Rolling Sharpe ratio
        max_drawdown: Max drawdown as negative decimal (e.g., -0.03 for -3%)

    Returns:
        Score 0-100
    """
    if len(daily_pnl_pcts) < 5:
        return 50.0  # Insufficient data

    try:
        # Component 1: % profitable days (40%)
        profitable = sum(1 for r in daily_pnl_pcts if r > 0)
        pct_positive = profitable / len(daily_pnl_pcts)

        # Component 2: Sharpe ratio normalized (30%)
        # Sharpe 2.0+ = perfect (1.0), Sharpe 0 = 0
        sharpe_norm = max(0.0, min(sharpe / 2.0, 1.0))

        # Component 3: Max drawdown normalized (30%)
        # 0% drawdown = perfect (1.0), -5%+ = 0
        dd_norm = max(0.0, 1.0 + max_drawdown / 0.05)

        score = pct_positive * 40 + sharpe_norm * 30 + dd_norm * 30
        return round(max(0.0, min(100.0, score)), 1)

    except Exception as e:
        logger.warning(f"Consistency score computation failed: {e}")
        return 50.0
