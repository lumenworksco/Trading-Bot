# Algo Trading Bot V2

Algorithmic trading bot using Alpaca API with three strategies: Opening Range Breakout (ORB), VWAP Mean Reversion, and Catalyst Momentum.

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

Paper mode by default. For live trading: `python main.py --live`

### 5. Run Backtest

```bash
python main.py --backtest
```

## V2 Features

### Three Strategies
- **ORB**: Trades breakouts above 30-min opening range. 3:1 R/R. Exits by 3:45 PM.
- **VWAP**: Buys at lower VWAP band with RSI confirmation. 45-min time stop.
- **Momentum**: Trades post-catalyst continuation. Holds 1-5 days. Max 1 position.

### 150 Symbol Universe
- 50 core liquid stocks/ETFs + 100 extended (growth, sector ETFs, mid-caps)
- Leveraged ETFs restricted to VWAP strategy only

### Smart Filters
- Earnings filter (skips trades within 48h of earnings)
- Correlation filter (skips if >75% correlated with open position)

### ATR-Based Position Sizing
- Risks 1% of portfolio per trade based on stop distance
- Hard cap: 6% per position. Bearish regime cuts size 40%.

### SQLite Database (bot.db)
- All trades, signals, daily snapshots, backtest results logged
- Open positions persisted (replaces state.json)

### Backtesting Engine
- 6 months hourly data via yfinance
- Slippage + commission simulation. Results saved to DB.

### Performance Dashboard
- Sharpe, Sortino, profit factor, max drawdown
- Strategy attribution, week P&L, earnings exclusion count

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | (required) | Alpaca API key |
| `ALPACA_API_SECRET` | (required) | Alpaca API secret |
| `ALPACA_LIVE` | `false` | `true` for live trading |
| `ALLOW_SHORT` | `false` | Enable short selling |
| `ALLOW_MOMENTUM` | `true` | Enable momentum strategy |

## File Structure

```
trading_bot/
├── main.py              # Entry point (--backtest, --live)
├── config.py            # Settings, 150 symbols, leveraged ETFs
├── database.py          # SQLite schema + queries
├── data.py              # Alpaca data fetching
├── strategies/          # Strategy package
│   ├── base.py          # Signal dataclass
│   ├── regime.py        # Market regime (SPY EMA)
│   ├── orb.py           # Opening Range Breakout
│   ├── vwap.py          # VWAP Mean Reversion
│   └── momentum.py      # Catalyst Momentum
├── backtester.py        # 6-month backtest engine
├── execution.py         # Order placement
├── risk.py              # ATR sizing + circuit breaker
├── earnings.py          # Earnings filter
├── correlation.py       # Correlation filter
├── analytics.py         # Sharpe, Sortino, drawdown
├── dashboard.py         # Rich terminal UI V2
├── start.sh / stop.sh / status.sh
├── trading_bot.service  # Systemd unit
└── requirements.txt
```

## Month 1 Testing Guide

**Week 1** (done): Validate basic mechanics run correctly.

**Week 2**: Run V2 paper mode. Monitor strategy attribution.

**Week 3**: Run backtest, compare to live paper results.

**Week 4**: If Sharpe > 0.8, win rate > 50% for 3 weeks, consider small live test ($1k max).

### Daily Checklist
- ORB signals: 3-8/day. VWAP: 5-15/day. Momentum: 0-2/week.
- Too many time-stops = entries too early.
- Check `./status.sh` and `bot.log` daily.
