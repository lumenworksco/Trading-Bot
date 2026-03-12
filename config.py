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

# Paper vs live base URLs
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
TRADE_SIZE_PCT = 0.03
DAILY_LOSS_HALT = -0.025
SCAN_INTERVAL_SEC = 60

# --- Strategy Filters ---
MAX_GAP_PCT = 0.03          # Skip ORB if gap > 3%
MAX_ORB_RANGE_PCT = 0.025   # Skip ORB if range > 2.5% of price
MAX_INTRADAY_MOVE_PCT = 0.03  # Skip VWAP if stock moved > 3% today

# --- ORB Settings ---
ORB_VOLUME_MULTIPLIER = 1.5   # Volume must be 1.5x average for breakout
ORB_TAKE_PROFIT_MULT = 1.5    # 1.5x ORB range above entry
ORB_STOP_LOSS_MULT = 0.5      # 0.5x ORB range below entry
ORB_TOP_N_SYMBOLS = 15        # Only trade top 15 by volume at 10:00 AM
ORB_ENTRY_SLIPPAGE = 0.0005   # Limit order at breakout + 0.05%

# --- VWAP Settings ---
VWAP_BAND_STD = 1.5           # VWAP band width in std devs
VWAP_RSI_OVERSOLD = 40        # RSI threshold for buy
VWAP_RSI_OVERBOUGHT = 60      # RSI threshold for sell/short
VWAP_STOP_EXTENSION = 0.5     # Stop at 0.5 std dev further against
VWAP_TIME_STOP_MINUTES = 45   # Exit after 45 min if no target/stop

# --- Market Regime ---
REGIME_CHECK_INTERVAL_MIN = 30  # Check SPY regime every 30 min
REGIME_EMA_PERIOD = 20          # 20-day EMA for SPY
BEARISH_SIZE_CUT = 0.40         # Cut sizes by 40% in bearish regime

# --- Market Hours (ET) ---
MARKET_OPEN = time(9, 30)
ORB_END = time(10, 0)
TRADING_START = time(10, 0)
ORB_EXIT_TIME = time(15, 45)
MARKET_CLOSE = time(16, 0)
EOD_SUMMARY_TIME = time(16, 15)

# --- Symbol Universe ---
SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL",
    "META", "NFLX", "AMD", "INTC", "SOFI", "PLTR", "COIN", "PYPL",
    "SNAP", "BABA", "JD", "SHOP", "JPM", "BAC", "GS", "V", "MA",
    "UNH", "CVX", "XOM", "LLY", "PFE", "SQ", "UBER", "LYFT", "ABNB",
    "RBLX", "AFRM", "HOOD", "CRWD", "PANW", "ZS", "IWM", "DIA",
    "XLF", "XLE", "XLK", "XLV", "ARKK", "GLD", "SLV", "TLT",
]

# --- State ---
STATE_FILE = "state.json"
LOG_FILE = "bot.log"
STATE_SAVE_INTERVAL_SEC = 60
