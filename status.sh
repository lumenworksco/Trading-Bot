#!/bin/bash
# Show bot status: uptime, last log lines, open positions
cd "$(dirname "$0")"

echo "=== ALGO TRADING BOT V2 STATUS ==="
echo

# Check if running
PID=$(pgrep -f "python3 main.py")
if [ -n "$PID" ]; then
    echo "Status:  RUNNING (PID $PID)"
    # Get uptime
    if [ -f /proc/$PID/stat ]; then
        UPTIME=$(ps -p "$PID" -o etime= 2>/dev/null)
        echo "Uptime:  $UPTIME"
    else
        echo "Uptime:  $(ps -p "$PID" -o etime= 2>/dev/null || echo 'unknown')"
    fi
else
    echo "Status:  STOPPED"
fi
echo

# Open positions from DB
if [ -f bot.db ]; then
    echo "--- Open Positions ---"
    sqlite3 bot.db "SELECT symbol, strategy, side, entry_price, qty, hold_type FROM open_positions;" 2>/dev/null || echo "(none)"
    echo
    echo "--- Today's Trades ---"
    TODAY=$(date +%Y-%m-%d)
    sqlite3 bot.db "SELECT COUNT(*) || ' trades today' FROM trades WHERE exit_time LIKE '${TODAY}%';" 2>/dev/null
    echo
fi

# Last log lines
if [ -f bot.log ]; then
    echo "--- Last 10 Log Lines ---"
    tail -10 bot.log
fi

echo
echo "==================================="
