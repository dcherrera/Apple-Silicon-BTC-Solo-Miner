#!/bin/bash
#
# BTC Solo Miner - Start Script
# Starts both the miner daemon and menu bar app
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting BTC Solo Miner..."

# Start the miner daemon
python3 "$SCRIPT_DIR/btc_miner.py" --daemon

# Give it a moment
sleep 2

# Start the menu bar app
python3 "$SCRIPT_DIR/miner_menubar.py" &

echo ""
echo "Done! Look for the ₿ icon in your menu bar."
echo ""
echo "To stop: Click ₿ → 'Stop Mining' or run: python3 btc_miner.py --stop"
