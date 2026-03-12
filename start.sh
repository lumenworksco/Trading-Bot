#!/bin/bash
# Start the trading bot in the background
cd "$(dirname "$0")"

if pgrep -f "python3 main.py" > /dev/null; then
    echo "Bot is already running."
    exit 1
fi

echo "Starting Algo Trading Bot V2..."
nohup python3 main.py "$@" > /dev/null 2>&1 &
echo "Bot started with PID $!"
echo $! > .bot.pid
