#!/usr/bin/env python3
"""
Bitcoin Miner Menu Bar App

A macOS menu bar application to monitor and control the Bitcoin solo miner.
Shows hashrate, current block, and total hashes in real-time.

Usage:
    python3 miner_menubar.py
"""

import rumps
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime

# Hide dock icon - must be done before any AppKit/rumps initialization
import AppKit
info = AppKit.NSBundle.mainBundle().infoDictionary()
info["LSUIElement"] = "1"

# Paths
CONFIG_DIR = Path.home() / '.btc_miner'
CONFIG_FILE = CONFIG_DIR / 'config.json'
STATS_FILE = CONFIG_DIR / 'stats.json'
PID_FILE = CONFIG_DIR / 'miner.pid'
LOG_FILE = CONFIG_DIR / 'miner.log'
WINS_FILE = CONFIG_DIR / 'wins.json'

MINER_SCRIPT = Path(__file__).parent / 'btc_miner.py'


def format_hashrate(h: float) -> str:
    """Format hashrate with SI prefix."""
    if h >= 1e12:
        return f"{h/1e12:.2f} TH/s"
    elif h >= 1e9:
        return f"{h/1e9:.2f} GH/s"
    elif h >= 1e6:
        return f"{h/1e6:.2f} MH/s"
    elif h >= 1e3:
        return f"{h/1e3:.2f} KH/s"
    else:
        return f"{h:.0f} H/s"


def format_number(n: int) -> str:
    """Format large numbers with suffixes."""
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    elif n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)


def format_time(seconds: float) -> str:
    """Format time duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    elif seconds < 86400 * 365:
        return f"{seconds/86400:.1f}d"
    else:
        return f"{seconds/(86400*365):.1f}y"


class BitcoinMinerApp(rumps.App):
    def __init__(self):
        super().__init__(
            "BTC",
            icon=None,
            template=True
        )

        # Menu items
        self.status_item = rumps.MenuItem("Status: Checking...")
        self.gpu_item = rumps.MenuItem("GPU: --")
        self.hashrate_item = rumps.MenuItem("Hashrate: --")
        self.block_item = rumps.MenuItem("Block: --")
        self.hashes_item = rumps.MenuItem("Hashes: --")
        self.uptime_item = rumps.MenuItem("Uptime: --")
        self.wallet_item = rumps.MenuItem("Wallet: --")

        self.menu = [
            self.status_item,
            self.gpu_item,
            None,  # Separator
            self.hashrate_item,
            self.block_item,
            self.hashes_item,
            self.uptime_item,
            None,  # Separator
            self.wallet_item,
            rumps.MenuItem("Set Wallet Address...", callback=self.set_wallet),
            None,  # Separator
            rumps.MenuItem("Start Mining", callback=self.start_mining),
            rumps.MenuItem("Stop Mining", callback=self.stop_mining),
            None,  # Separator
            rumps.MenuItem("View Logs", callback=self.view_logs),
            rumps.MenuItem("Open Folder", callback=self.open_folder),
            None,  # Separator
            rumps.MenuItem("Donate to Developer", callback=self.show_donate),
            None,  # Separator
        ]

        # Start update timer
        self.timer = rumps.Timer(self.update_status, 2)
        self.timer.start()

        # Check for wins periodically
        self.win_check_timer = rumps.Timer(self.check_wins, 10)
        self.win_check_timer.start()

        self.last_win_count = 0

    def is_running(self) -> bool:
        """Check if miner daemon is running."""
        if not PID_FILE.exists():
            return False

        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError):
            return False

    def get_stats(self) -> dict:
        """Get current miner stats."""
        if not STATS_FILE.exists():
            return {}

        try:
            return json.loads(STATS_FILE.read_text())
        except:
            return {}

    def get_config(self) -> dict:
        """Get miner config."""
        if not CONFIG_FILE.exists():
            return {}

        try:
            return json.loads(CONFIG_FILE.read_text())
        except:
            return {}

    def update_status(self, _):
        """Update menu bar status."""
        running = self.is_running()
        stats = self.get_stats()
        config = self.get_config()

        # GPU status
        gpu_enabled = stats.get('gpu_enabled', False)
        power_mode = stats.get('power_mode', 'AC')
        self.gpu_item.title = f"GPU: {'Apple M4' if gpu_enabled else 'Disabled (CPU only)'} [{power_mode}]"

        if running:
            hashrate = stats.get('hashrate', 0)

            # Update title with hashrate and power indicator
            if hashrate > 0:
                if power_mode == 'Battery':
                    power_icon = "🔋"  # Battery mode - throttled
                else:
                    power_icon = "⚡" if gpu_enabled else ""  # AC power - full speed
                self.title = f"₿{power_icon} {format_hashrate(hashrate)}"
            else:
                self.title = "₿ Mining..."

            self.status_item.title = f"Status: Running {'(GPU)' if gpu_enabled else '(CPU)'} [{power_mode}]"
            self.hashrate_item.title = f"Hashrate: {format_hashrate(hashrate)}"
            self.block_item.title = f"Block: {stats.get('current_height', '--')}"
            self.hashes_item.title = f"Hashes: {format_number(stats.get('total_hashes', 0))}"

            # Calculate uptime
            start_time = stats.get('start_time')
            if start_time:
                try:
                    start_dt = datetime.fromisoformat(start_time)
                    uptime = (datetime.now() - start_dt).total_seconds()
                    self.uptime_item.title = f"Uptime: {format_time(uptime)}"
                except:
                    self.uptime_item.title = "Uptime: --"
            else:
                self.uptime_item.title = "Uptime: --"

        else:
            self.title = "₿ Off"
            self.status_item.title = "Status: Stopped"
            self.hashrate_item.title = "Hashrate: --"
            self.block_item.title = "Block: --"
            self.hashes_item.title = f"Total Hashes: {format_number(stats.get('total_hashes', 0))}"
            self.uptime_item.title = "Uptime: --"

        # Wallet
        wallet = config.get('wallet_address', 'Not Set')
        if wallet:
            self.wallet_item.title = f"Wallet: {wallet[:8]}...{wallet[-4:]}"
        else:
            self.wallet_item.title = "Wallet: Not Set"

    def check_wins(self, _):
        """Check for new block wins."""
        if not WINS_FILE.exists():
            return

        try:
            wins = json.loads(WINS_FILE.read_text())
            if len(wins) > self.last_win_count:
                # New win!
                latest = wins[-1]
                rumps.notification(
                    title="BITCOIN BLOCK FOUND!",
                    subtitle=f"Block {latest['height']}",
                    message=f"Hash: {latest['hash'][:32]}...\nCheck wins.json for details!",
                    sound=True
                )
                self.last_win_count = len(wins)
        except:
            pass

    def start_mining(self, _):
        """Start the miner daemon."""
        if self.is_running():
            rumps.notification(
                title="Bitcoin Miner",
                subtitle="Already Running",
                message="The miner is already running."
            )
            return

        try:
            subprocess.run(
                ['python3', str(MINER_SCRIPT), '--daemon'],
                capture_output=True
            )
            time.sleep(1)

            if self.is_running():
                rumps.notification(
                    title="Bitcoin Miner",
                    subtitle="Started",
                    message="Mining has started in the background."
                )
            else:
                rumps.notification(
                    title="Bitcoin Miner",
                    subtitle="Failed to Start",
                    message="Check the logs for errors."
                )
        except Exception as e:
            rumps.notification(
                title="Bitcoin Miner",
                subtitle="Error",
                message=str(e)
            )

    def stop_mining(self, _):
        """Stop the miner daemon."""
        if not self.is_running():
            rumps.notification(
                title="Bitcoin Miner",
                subtitle="Not Running",
                message="The miner is not currently running."
            )
            return

        try:
            subprocess.run(
                ['python3', str(MINER_SCRIPT), '--stop'],
                capture_output=True
            )
            time.sleep(1)

            if not self.is_running():
                rumps.notification(
                    title="Bitcoin Miner",
                    subtitle="Stopped",
                    message="Mining has been stopped."
                )
            else:
                rumps.notification(
                    title="Bitcoin Miner",
                    subtitle="Warning",
                    message="Miner may still be running."
                )
        except Exception as e:
            rumps.notification(
                title="Bitcoin Miner",
                subtitle="Error",
                message=str(e)
            )

    def view_logs(self, _):
        """Open log file in Console."""
        if LOG_FILE.exists():
            subprocess.run(['open', '-a', 'Console', str(LOG_FILE)])
        else:
            rumps.notification(
                title="Bitcoin Miner",
                subtitle="No Logs",
                message="Log file not found."
            )

    def open_folder(self, _):
        """Open miner config folder in Finder."""
        CONFIG_DIR.mkdir(exist_ok=True)
        subprocess.run(['open', str(CONFIG_DIR)])

    def set_wallet(self, _):
        """Prompt user to set wallet address."""
        # Get current wallet
        config = self.get_config()
        current_wallet = config.get('wallet_address', '')

        # Show input dialog
        response = rumps.Window(
            title="Set Bitcoin Wallet Address",
            message="Enter your Bitcoin wallet address to receive block rewards.\n\nSupported formats: bc1q... (SegWit), 1... (Legacy), 3... (P2SH)",
            default_text=current_wallet or "",
            ok="Save",
            cancel="Cancel",
            dimensions=(400, 24)
        ).run()

        if response.clicked:
            new_wallet = response.text.strip()

            if new_wallet:
                # Basic validation
                if not (new_wallet.startswith('bc1') or
                        new_wallet.startswith('1') or
                        new_wallet.startswith('3')):
                    rumps.notification(
                        title="Invalid Address",
                        subtitle="Address Format Error",
                        message="Bitcoin addresses should start with 'bc1', '1', or '3'"
                    )
                    return

                # Save to config
                config['wallet_address'] = new_wallet
                self.save_config(config)

                rumps.notification(
                    title="Wallet Updated",
                    subtitle="Address Saved",
                    message=f"Set to: {new_wallet[:12]}...{new_wallet[-4:]}"
                )

                # Restart miner if running to pick up new address
                if self.is_running():
                    rumps.notification(
                        title="Restart Required",
                        subtitle="",
                        message="Restart the miner for the new address to take effect."
                    )
            else:
                # Clear wallet
                config['wallet_address'] = None
                self.save_config(config)
                rumps.notification(
                    title="Wallet Cleared",
                    subtitle="",
                    message="No wallet address set. Block rewards will be lost!"
                )

    def save_config(self, config: dict):
        """Save config to file."""
        CONFIG_DIR.mkdir(exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(config, indent=2))

    def show_donate(self, _):
        """Show donation address."""
        donate_address = "bc1qvaehlavm6w3tygf8rfmkea9keumv3mzt4y8and"

        response = rumps.Window(
            title="Support the Developer",
            message="If you find this miner useful, consider sending a tip!\n\n"
                    "Bitcoin Address (click to copy):\n\n"
                    f"{donate_address}\n\n"
                    "Every satoshi helps! Thank you for your support.",
            default_text=donate_address,
            ok="Copy Address",
            cancel="Close",
            dimensions=(420, 24)
        ).run()

        if response.clicked:
            # Copy to clipboard
            subprocess.run(['pbcopy'], input=donate_address.encode(), check=True)
            rumps.notification(
                title="Address Copied!",
                subtitle="",
                message="Bitcoin address copied to clipboard. Thank you!"
            )


if __name__ == '__main__':
    BitcoinMinerApp().run()
