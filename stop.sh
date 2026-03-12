#!/bin/bash
# Gracefully stop the trading bot (sends SIGINT for clean shutdown)
cd "$(dirname "$0")"

if [ -f .bot.pid ]; then
    PID=$(cat .bot.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping bot (PID $PID)..."
        kill -INT "$PID"
        # Wait up to 30 seconds for graceful shutdown
        for i in $(seq 1 30); do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "Bot stopped gracefully."
                rm -f .bot.pid
                exit 0
            fi
            sleep 1
        done
        echo "Bot did not stop in 30s, forcing..."
        kill -9 "$PID"
        rm -f .bot.pid
    else
        echo "Bot is not running (stale PID file)."
        rm -f .bot.pid
    fi
else
    # Try to find by process name
    PID=$(pgrep -f "python3 main.py")
    if [ -n "$PID" ]; then
        echo "Stopping bot (PID $PID)..."
        kill -INT "$PID"
        sleep 5
        echo "Bot stopped."
    else
        echo "Bot is not running."
    fi
fi
