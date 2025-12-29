#!/bin/bash
#
# BTC Solo Miner - Install Script
# For macOS with Apple Silicon (M1/M2/M3/M4)
#

set -e

echo "=============================================="
echo "  BTC Solo Miner - Installer"
echo "=============================================="
echo ""

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: This miner is optimized for Apple Silicon (M1/M2/M3/M4)."
    echo "It will still work on Intel Macs but without GPU acceleration."
    echo ""
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Install it from: https://www.python.org/downloads/"
    exit 1
fi

echo "Installing Python dependencies..."
pip3 install --user rumps requests pyobjc-framework-Metal

echo ""
echo "Installation complete!"
echo ""
echo "=============================================="
echo "  Quick Start"
echo "=============================================="
echo ""
echo "1. Start the menu bar app:"
echo "   python3 $(dirname "$0")/miner_menubar.py &"
echo ""
echo "2. Click the ₿ icon in your menu bar"
echo ""
echo "3. Set your wallet address:"
echo "   Menu → 'Set Wallet Address...'"
echo ""
echo "4. Start mining:"
echo "   Menu → 'Start Mining'"
echo ""
echo "=============================================="
echo ""
echo "Want to start mining right now? Run:"
echo "   $(dirname "$0")/start.sh"
echo ""
