"""Rich terminal dashboard — updates every 5 seconds."""

import logging
from datetime import datetime, timedelta

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
from risk import RiskManager, TradeRecord

logger = logging.getLogger(__name__)
console = Console()


def format_duration(start: datetime, now: datetime) -> str:
    delta = now - start
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes, _ = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def format_pnl(pnl: float) -> str:
    if pnl >= 0:
        return f"[green]+${pnl:.0f}[/green]"
    return f"[red]-${abs(pnl):.0f}[/red]"


def format_pnl_pct(pct: float) -> str:
    if pct >= 0:
        return f"[green]+{pct:.1%}[/green]"
    return f"[red]{pct:.1%}[/red]"


def build_dashboard(
    risk: RiskManager,
    regime: str,
    start_time: datetime,
    now: datetime,
    last_scan_time: datetime | None,
    num_symbols: int,
) -> Panel:
    """Build the full dashboard layout as a Rich Panel."""

    mode = "PAPER" if config.PAPER_MODE else "[bold red]LIVE[/bold red]"
    uptime = format_duration(start_time, now)

    regime_color = "green" if regime == "BULLISH" else "red" if regime == "BEARISH" else "yellow"
    regime_text = f"[{regime_color}]{regime}[/{regime_color}]"

    # Circuit breaker
    cb_text = (
        "[bold red]ACTIVE - NO NEW TRADES[/bold red]"
        if risk.circuit_breaker_active
        else "[green]INACTIVE[/green]"
    )

    # Header
    header = (
        f"  ALGO BOT | {mode} MODE | Regime: {regime_text} | Up: {uptime}"
    )

    # Portfolio section
    day_pnl_dollars = risk.day_pnl * risk.starting_equity if risk.starting_equity else 0
    portfolio_lines = [
        f"  Value:   ${risk.current_equity:,.0f}",
        f"  Cash:    ${risk.current_cash:,.0f}",
        f"  Day P&L: {format_pnl(day_pnl_dollars)} {format_pnl_pct(risk.day_pnl)}",
    ]

    # Open positions
    open_count = len(risk.open_trades)
    pos_header = f"OPEN POSITIONS ({open_count}/{config.MAX_POSITIONS})"

    pos_lines = []
    for symbol, trade in risk.open_trades.items():
        held_time = format_duration(trade.entry_time, now)
        # Estimate unrealized P&L (we don't have live price here, show entry info)
        pos_lines.append(
            f"  {symbol:<6} {trade.strategy:<5} "
            f"entry=${trade.entry_price:<8.2f} "
            f"qty={trade.qty:<4} {held_time}"
        )

    if not pos_lines:
        pos_lines = ["  (none)"]

    # Recent trades
    recent = risk.closed_trades[-6:] if risk.closed_trades else []
    recent_lines = []
    for trade in reversed(recent):
        icon = "[green]W[/green]" if trade.pnl > 0 else "[red]L[/red]"
        time_str = trade.exit_time.strftime("%H:%M") if trade.exit_time else "??:??"
        pnl_pct = trade.pnl / (trade.entry_price * trade.qty) if trade.entry_price * trade.qty > 0 else 0
        recent_lines.append(
            f"  {time_str} {trade.side.upper():<5} {trade.symbol:<6} "
            f"{trade.strategy:<5} {format_pnl(trade.pnl)} "
            f"{format_pnl_pct(pnl_pct)} {icon}"
        )
    if not recent_lines:
        recent_lines = ["  (no trades yet)"]

    # Footer
    scan_time_str = last_scan_time.strftime("%H:%M:%S") if last_scan_time else "---"
    footer = (
        f"  Last scan: {scan_time_str} | "
        f"{num_symbols} symbols | "
        f"Signals today: {risk.signals_today}\n"
        f"  Circuit breaker: {cb_text}"
    )

    # Assemble
    content = (
        f"[bold]{header}[/bold]\n"
        f"{'─' * 60}\n"
        f" PORTFOLIO\n"
        + "\n".join(portfolio_lines)
        + f"\n{'─' * 60}\n"
        f" {pos_header}\n"
        + "\n".join(pos_lines)
        + f"\n{'─' * 60}\n"
        f" RECENT TRADES\n"
        + "\n".join(recent_lines)
        + f"\n{'─' * 60}\n"
        + footer
    )

    return Panel(content, title="[bold cyan]ALGO TRADING BOT[/bold cyan]", border_style="cyan")


def print_day_summary(summary: dict):
    """Print the end-of-day summary."""
    if summary.get("trades", 0) == 0:
        console.print("\n[yellow]=== DAY SUMMARY ===[/yellow]")
        console.print("  No trades today.")
        console.print("[yellow]===================[/yellow]\n")
        return

    console.print("\n[bold yellow]=== DAY SUMMARY ===[/bold yellow]")
    console.print(f"  Trades today:     {summary['trades']}")
    console.print(f"  Winners:          {summary['winners']}  ({summary['win_rate']:.0%})")
    console.print(f"  Losers:           {summary['losers']}  ({1 - summary['win_rate']:.0%})")
    console.print(f"  Day P&L:         ${summary['total_pnl']:+.0f} ({summary['pnl_pct']:+.1%})")
    console.print(f"  Best trade:       {summary['best_trade']}")
    console.print(f"  Worst trade:      {summary['worst_trade']}")
    console.print(f"  ORB win rate:     {summary['orb_win_rate']}")
    console.print(f"  VWAP win rate:    {summary['vwap_win_rate']}")
    console.print("[bold yellow]===================[/bold yellow]\n")


def print_startup_info(info: dict):
    """Print startup connectivity info."""
    mode = "PAPER" if info["paper"] else "[bold red]LIVE[/bold red]"
    market = "[green]OPEN[/green]" if info["market_open"] else "[yellow]CLOSED[/yellow]"

    console.print(Panel(
        f"  Account: {info['account_id']}\n"
        f"  Mode:    {mode}\n"
        f"  Equity:  ${info['equity']:,.2f}\n"
        f"  Cash:    ${info['cash']:,.2f}\n"
        f"  Market:  {market}\n"
        f"  Next:    {info.get('next_open', 'N/A')}",
        title="[bold green]CONNECTION VERIFIED[/bold green]",
        border_style="green",
    ))
