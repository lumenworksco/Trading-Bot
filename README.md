# Algo Trading Bot

Algorithmic trading bot using Alpaca API with two strategies: Opening Range Breakout (ORB) and VWAP Mean Reversion.

## Quick Start (< 5 minutes)

### 1. Get Alpaca API Keys

Sign up at [alpaca.markets](https://alpaca.markets) and create a paper trading API key.

### 2. Install Dependencies

```bash
cd trading_bot
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_API_SECRET="your-api-secret"
```

### 4. Run

```bash
python main.py
```

The bot starts in **paper mode** by default. To trade live (use with caution):

```bash
export ALPACA_LIVE=true
```

## Configuration

All settings are in `config.py` or controllable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | (required) | Your Alpaca API key |
| `ALPACA_API_SECRET` | (required) | Your Alpaca API secret |
| `ALPACA_LIVE` | `false` | Set to `true` for live trading |
| `ALLOW_SHORT` | `false` | Set to `true` to enable short selling |

## Strategies

- **ORB (Opening Range Breakout)**: Trades breakouts above the first 30-minute range. Runs 10:00 AM - 3:45 PM ET. 3:1 risk/reward.
- **VWAP Mean Reversion**: Buys at lower VWAP band with RSI confirmation, targets VWAP. 45-minute time stop.

## Risk Management

- 3% of cash per trade
- Max 6 open positions
- Max 25% portfolio deployed
- Daily loss halt at -2.5%
- Every order is a bracket order (entry + TP + SL)

## File Structure

```
trading_bot/
├── main.py          # Entry point + main loop
├── config.py        # All settings
├── strategies.py    # ORB and VWAP strategies
├── execution.py     # Order placement
├── risk.py          # Position sizing + circuit breaker
├── data.py          # Alpaca data fetching
├── dashboard.py     # Rich terminal UI
├── requirements.txt
└── README.md
```
