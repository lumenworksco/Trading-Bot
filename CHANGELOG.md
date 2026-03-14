# Changelog

All notable changes to this project will be documented in this file.

## [7.0.0] - 2026-03-14

### Bug Fixes
- **Broker sync 0 P&L** — `sync_positions_with_broker()` now fetches market price via `get_snapshot()` instead of using entry_price as exit_price
- **StatMR never firing** — Added startup initialization for `prepare_universe()` + switched to 2-min intraday bars for OU fitting (correct half-life conversion)
- **KalmanPairs never initializing** — Loads existing pairs from DB on startup; runs `select_pairs_weekly()` if table is empty
- **MTF over-filtering** — Per-strategy MTF control via `MTF_ENABLED_FOR` config dict; disabled for mean reversion strategies

### Added
- **VWAP v2 Hybrid strategy** (20% allocation) — VWAP + OU z-score dual confirmation with bid-ask spread filter
- **ORB v2 strategy** (5% allocation) — Opening range breakout 10:00–11:30 AM with gap/range quality filters
- **Alpaca News Sentiment** — Keyword-based headline scoring, soft position-size multiplier, 30-min cache
- **LLM Signal Scoring** — Optional Claude Haiku signal evaluation; fail-open, 3s timeout, $0.10/day cost cap
- **Adaptive VIX-Aware Exits** — Four VIX regimes with dynamic exit parameters
- **Walk-Forward Validation** — Weekly OOS Sharpe check per strategy with auto-demotion
- **Strategy Health Dashboard** — `/api/strategy_health` and `/api/filter_diagnostic` endpoints
- 196 unit tests (up from ~108)

### Changed
- Strategy allocations: StatMR 50%, VWAP 20%, Pairs 20%, ORB 5%, MicroMom 5%
- Notifications switched from WhatsApp to Telegram Bot API
- Dropped Python 3.11 support; minimum is now 3.12
- Dockerfile updated to Python 3.13
- Dashboard and web UI updated to V7 branding with new strategy filters

### Dependencies
- Added `anthropic>=0.49.0` (optional, for LLM scoring)

## [6.0.0] - 2026-03-13

### Added
- Complete rebuild around statistical mean reversion
- StatMeanReversion (60%), KalmanPairsTrader (25%), IntradayMicroMomentum (15%)
- Volatility-targeted position sizing (1% daily vol target)
- Daily P&L locking (GAIN_LOCK at +1.5%, LOSS_HALT at -1.0%)
- Beta neutralization with SPY hedging
- OU parameter fitting, Hurst exponent, consistency scoring analytics
- TWAP order execution for large orders

## [4.0.0] - 2026-03-13

### Added
- Multi-timeframe (MTF) confirmation for trade entries
- VIX-based risk scaling to reduce exposure in high-volatility regimes
- News sentiment filter via API integration
- Sector rotation strategy using ETF relative strength
- Pairs trading strategy with cointegration detection
- Advanced exit types: scaled take-profits, trailing stops, RSI-based exits, volatility-based exits
- Docker and docker-compose deployment
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions and Codecov

## [3.0.0] - 2025-09-01

### Added
- ML-based signal filter for trade quality scoring
- Dynamic capital allocation across strategies
- WebSocket real-time price monitoring
- Short selling support
- Gap & Go strategy (pre-market gap continuation)
- Relative strength scanning
- Telegram alerts for fills, errors, and daily P&L summaries
- Web dashboard with live positions and equity curve
- Auto-optimization of strategy parameters via walk-forward analysis

## [2.0.0] - 2025-05-01

### Added
- Momentum strategy for multi-day trend following
- SQLite database for trade logging and analytics
- Backtesting engine with historical data replay
- Earnings date filter to avoid holding through reports
- Correlation filter to limit exposure to correlated positions

## [1.0.0] - 2025-01-01

### Added
- Opening Range Breakout (ORB) strategy with 3:1 R/R
- VWAP Mean Reversion strategy with 45-minute time stop
- Bracket order execution (entry + take-profit + stop-loss)
- Market regime detection via SPY 20-day EMA
- Rich terminal dashboard with live display
- State persistence via state.json
- Circuit breaker at -2.5% daily loss
- Paper/live mode switching via ALPACA_LIVE environment variable
- 50 hardcoded liquid symbols
