"""Configuration — all settings and environment variable handling."""

import os
from datetime import time
from zoneinfo import ZoneInfo

# --- Timezone ---
ET = ZoneInfo("America/New_York")

# --- Alpaca API ---
PAPER_MODE = os.getenv("ALPACA_LIVE", "false") != "true"
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

BASE_URL = (
    "https://paper-api.alpaca.markets"
    if PAPER_MODE
    else "https://api.alpaca.markets"
)
DATA_URL = "https://data.alpaca.markets"

# --- Trading ---
ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false") == "true"
MAX_POSITIONS = 6
MAX_PORTFOLIO_DEPLOY = 0.25
TRADE_SIZE_PCT = 0.03               # Legacy flat sizing (kept for reference)
DAILY_LOSS_HALT = -0.025
SCAN_INTERVAL_SEC = 60

# --- V2: Position Sizing (ATR-based) ---
RISK_PER_TRADE_PCT = 0.01           # Risk 1% of portfolio per trade
MAX_POSITION_PCT = 0.06             # Hard cap: max 6% of portfolio per position
MIN_POSITION_VALUE = 100             # Min $100 per trade

# --- Strategy Filters ---
MAX_GAP_PCT = 0.03                  # Skip ORB if gap > 3%
MAX_ORB_RANGE_PCT = 0.025           # Skip ORB if range > 2.5% of price
MAX_INTRADAY_MOVE_PCT = 0.03        # Skip VWAP if stock moved > 3% today

# --- ORB Settings ---
ORB_VOLUME_MULTIPLIER = 1.5
ORB_TAKE_PROFIT_MULT = 1.5
ORB_STOP_LOSS_MULT = 0.5
ORB_TOP_N_SYMBOLS = 15
ORB_ENTRY_SLIPPAGE = 0.0005

# --- VWAP Settings ---
VWAP_BAND_STD = 1.5
VWAP_RSI_OVERSOLD = 40
VWAP_RSI_OVERBOUGHT = 60
VWAP_STOP_EXTENSION = 0.5
VWAP_TIME_STOP_MINUTES = 45

# --- V2: Momentum Settings ---
ALLOW_MOMENTUM = os.getenv("ALLOW_MOMENTUM", "true") == "true"
MAX_MOMENTUM_POSITIONS = 1          # Only 1 swing position at once
MOMENTUM_MIN_MOVE_PCT = 0.04        # Yesterday must move > +4%
MOMENTUM_VOL_MULTIPLIER = 2.0       # Volume > 2x 30-day average
MOMENTUM_CONSOLIDATION_PCT = 0.015  # Today within 1.5% of yesterday's close
MOMENTUM_MAX_STOP_PCT = 0.02        # Max -2% stop from entry
MOMENTUM_TP1_PCT = 0.03             # Sell 50% at +3%
MOMENTUM_TP2_PCT = 0.06             # Sell remaining at +6%
MOMENTUM_TRAILING_STOP_PCT = 0.015  # 1.5% trailing stop
MOMENTUM_MAX_HOLD_DAYS = 5          # Exit on day 5 regardless
MOMENTUM_SCAN_TIME = time(10, 30)   # Scan once at 10:30 AM

# --- Market Regime ---
REGIME_CHECK_INTERVAL_MIN = 30
REGIME_EMA_PERIOD = 20
BEARISH_SIZE_CUT = 0.40

# --- V2: Filters ---
EARNINGS_FILTER_DAYS = 2            # Skip symbols with earnings within 2 days
CORRELATION_THRESHOLD = 0.75        # Skip if correlated > 75% with open position

# --- Market Hours (ET) ---
MARKET_OPEN = time(9, 30)
ORB_END = time(10, 0)
TRADING_START = time(10, 0)
ORB_EXIT_TIME = time(15, 45)
MARKET_CLOSE = time(16, 0)
EOD_SUMMARY_TIME = time(16, 15)

# --- Core Symbol Universe (original 50) ---
CORE_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL",
    "META", "NFLX", "AMD", "INTC", "SOFI", "PLTR", "COIN", "PYPL",
    "SNAP", "BABA", "JD", "SHOP", "JPM", "BAC", "GS", "V", "MA",
    "UNH", "CVX", "XOM", "LLY", "PFE", "SQ", "UBER", "LYFT", "ABNB",
    "RBLX", "AFRM", "HOOD", "CRWD", "PANW", "ZS", "IWM", "DIA",
    "XLF", "XLE", "XLK", "XLV", "ARKK", "GLD", "SLV", "TLT",
]

# --- V2: Extended Universe (+100) ---
LARGE_CAP_GROWTH = [
    "ADBE", "CRM", "NOW", "SNOW", "DDOG", "NET", "FTNT", "OKTA",
    "ZM", "DOCU", "TWLO", "MDB", "ESTC", "CFLT", "GTLB", "BILL",
    "HUBS", "VEEV", "WDAY", "ANSS", "TTD", "ROKU", "PINS", "ETSY",
    "W", "CHWY", "DASH", "APP",
]

SECTOR_ETFS = [
    "XLB", "XLI", "XLU", "XLRE", "XLC", "XLP", "XLY", "SOXX",
    "SMH", "IBB", "SOXL", "TQQQ", "SPXL", "UVXY", "SQQQ",
    "TNA", "FAS", "FAZ", "LABU", "LABD",
]

HIGH_MOMENTUM_MIDCAPS = [
    "CELH", "SMCI", "AXON", "PODD", "ENPH", "FSLR", "RUN", "BLNK",
    "CHPT", "BE", "IONQ", "RGTI", "QUBT", "ARQQ", "BBAI", "SOUN",
    "ASTS", "RDW", "RKLB", "LUNR", "DUOL", "CAVA", "BROS", "SHAK",
    "WING", "TXRH", "CMG", "DPZ", "DNUT", "JACK",
]

# Full universe
SYMBOLS = CORE_SYMBOLS + LARGE_CAP_GROWTH + SECTOR_ETFS + HIGH_MOMENTUM_MIDCAPS

# Leveraged ETFs — ONLY use VWAP on these, never ORB or Momentum
LEVERAGED_ETFS = {
    "SOXL", "TQQQ", "SPXL", "UVXY", "SQQQ", "TNA", "FAS", "FAZ", "LABU", "LABD",
}

# Non-leveraged symbols for ORB and Momentum
STANDARD_SYMBOLS = [s for s in SYMBOLS if s not in LEVERAGED_ETFS]

# --- State & Persistence ---
STATE_FILE = "state.json"
DB_FILE = "bot.db"
LOG_FILE = "bot.log"
STATE_SAVE_INTERVAL_SEC = 60

# --- Backtest ---
BACKTEST_SLIPPAGE = 0.0005          # 0.05% slippage per trade
BACKTEST_COMMISSION = 0.0035        # $0.0035 per share
BACKTEST_RISK_FREE_RATE = 0.045     # 4.5% annual
BACKTEST_TOP_N = 20                 # Run on top 20 most liquid symbols
