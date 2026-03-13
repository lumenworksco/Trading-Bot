"""Strategy package — V6 strategy exports + backward compatibility for archived strategies."""

from strategies.base import Signal, ORBRange, VWAPState
from strategies.regime import MarketRegime

# V6 strategies
from strategies.stat_mean_reversion import StatMeanReversion
from strategies.kalman_pairs import KalmanPairsTrader
from strategies.micro_momentum import IntradayMicroMomentum

# Backward compatibility: archived V1-V5 strategies (import only if needed)
try:
    from strategies.archive.orb import ORBStrategy
except ImportError:
    ORBStrategy = None

try:
    from strategies.archive.vwap import VWAPStrategy
except ImportError:
    VWAPStrategy = None

try:
    from strategies.archive.momentum import MomentumStrategy
except ImportError:
    MomentumStrategy = None

try:
    from strategies.archive.gap_go import GapGoStrategy
except ImportError:
    GapGoStrategy = None

try:
    from strategies.archive.sector_rotation import SectorRotationStrategy
except ImportError:
    SectorRotationStrategy = None

try:
    from strategies.archive.pairs_trading import PairsTradingStrategy
except ImportError:
    PairsTradingStrategy = None

try:
    from strategies.archive.momentum_scalp import EMAScalper
except ImportError:
    EMAScalper = None

__all__ = [
    # Shared types
    "Signal",
    "ORBRange",
    "VWAPState",
    "MarketRegime",
    # V6 strategies
    "StatMeanReversion",
    "KalmanPairsTrader",
    "IntradayMicroMomentum",
    # Archived (backward compat)
    "ORBStrategy",
    "VWAPStrategy",
    "MomentumStrategy",
    "GapGoStrategy",
    "SectorRotationStrategy",
    "PairsTradingStrategy",
    "EMAScalper",
]
