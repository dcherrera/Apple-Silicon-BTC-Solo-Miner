# Apple Silicon BTC Solo Miner

A Bitcoin solo miner for Apple Silicon Macs with GPU acceleration via Metal.

## Features

- **GPU Mining** - Uses Apple Metal for ~50-100 MH/s on M-series chips
- **Menu Bar App** - Monitor hashrate, block height, and stats from your menu bar
- **Power Aware** - Automatically throttles when on battery to save power
- **No Dock Icon** - Runs quietly in the background
- **Solo Mining** - Mine directly against the Bitcoin network (no pool)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+

## Quick Start (Easy Mode)

```bash
# 1. Clone and enter the folder
git clone https://github.com/dcherrera/Apple-Silicon-BTC-Solo-Miner.git
cd Apple-Silicon-BTC-Solo-Miner

# 2. Run the installer
./install.sh

# 3. Start mining!
./start.sh
```

That's it! Look for the **₿** icon in your menu bar.

## First Time Setup

1. Click the **₿** icon in your menu bar
2. Select **"Set Wallet Address..."**
3. Enter your Bitcoin wallet address (bc1q..., 1..., or 3...)
4. Click **"Start Mining"**

## Manual Installation

If you prefer to install manually:

```bash
# Install dependencies
pip3 install --user rumps requests pyobjc-framework-Metal

# Start the menu bar app
python3 miner_menubar.py &
```

## CLI Usage

```bash
# Set wallet address
python3 btc_miner.py --wallet bc1qYOURADDRESS

# Start mining daemon
python3 btc_miner.py --daemon

# Check status
python3 btc_miner.py --status

# Stop mining
python3 btc_miner.py --stop

# Run in foreground (for testing)
python3 btc_miner.py --foreground
```

## Power Modes

The miner automatically detects power status:

| Mode | Batch Size | Sleep | Hashrate |
|------|------------|-------|----------|
| AC Power (⚡) | 4M | 0ms | ~50-100 MH/s |
| Battery (🔋) | 256K | 100ms | ~2-5 MH/s |

## Files

- `btc_miner.py` - Main miner daemon
- `gpu_miner.py` - Metal GPU acceleration
- `miner_menubar.py` - macOS menu bar app
- `~/.btc_miner/` - Config, stats, logs

## Realistic Expectations

At 100 MH/s vs the Bitcoin network's ~700 EH/s, your odds of finding a block are approximately:

- **1 in 7 trillion** per hash
- **Expected time to find a block:** ~200,000 years

But someone has to find each block. It might as well be you. Good luck!

## Support

If you find this miner useful, you can send a tip via the menu bar:
**₿ → "Donate to Developer"**

Or send directly to: `bc1qvaehlavm6w3tygf8rfmkea9keumv3mzt4y8and`

## License

MIT
