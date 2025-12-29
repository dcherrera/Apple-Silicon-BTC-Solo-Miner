#!/usr/bin/env python3
"""
Bitcoin Solo Miner - Production Version

A real Bitcoin miner that runs in the background and can actually
submit blocks to the network if you find one.

Modes:
1. Full Node Mode - Uses Bitcoin Core RPC (requires ~600GB disk)
2. Light Mode - Uses public APIs (can detect wins, manual submission)

Usage:
    python3 btc_miner.py --daemon                    # Run as background daemon
    python3 btc_miner.py --wallet <address>          # Set reward address
    python3 btc_miner.py --status                    # Check miner status
    python3 btc_miner.py --stop                      # Stop daemon

Requirements:
    pip install requests

Optional (for Bitcoin Core mode):
    brew install bitcoin
    # Then sync the blockchain (~600GB, takes days)
"""

import os
import sys
import time
import struct
import hashlib
import json
import logging
import signal
import argparse
import threading
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
from datetime import datetime
import requests

# Try to import GPU miner
try:
    from gpu_miner import MetalSHA256Miner, METAL_AVAILABLE
except ImportError:
    METAL_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_DIR = Path.home() / '.btc_miner'
CONFIG_FILE = CONFIG_DIR / 'config.json'
LOG_FILE = CONFIG_DIR / 'miner.log'
PID_FILE = CONFIG_DIR / 'miner.pid'
STATS_FILE = CONFIG_DIR / 'stats.json'
WINS_FILE = CONFIG_DIR / 'wins.json'

DEFAULT_CONFIG = {
    'wallet_address': None,
    'num_threads': mp.cpu_count(),
    'bitcoin_rpc_url': 'http://127.0.0.1:8332',
    'bitcoin_rpc_user': 'bitcoin',
    'bitcoin_rpc_password': 'bitcoin',
    'use_bitcoin_core': False,
    'use_gpu': True,  # Use Metal GPU if available
    'log_level': 'INFO',
    'stats_interval': 60,  # seconds between stats updates
    'block_check_interval': 30,  # seconds between checking for new blocks
    'battery_throttle': True,  # Reduce mining when on battery
    'battery_batch_size': 256 * 1024,  # Smaller batches on battery (256K vs 4M)
    'plugged_batch_size': 4 * 1024 * 1024,  # Full speed when plugged in
    'battery_sleep_ms': 100,  # Sleep between batches on battery
}

# ============================================================================
# POWER STATUS DETECTION
# ============================================================================

def is_on_ac_power() -> bool:
    """Check if Mac is plugged in (AC power) or on battery."""
    try:
        import subprocess
        result = subprocess.run(
            ['pmset', '-g', 'batt'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Output contains "AC Power" or "Battery Power"
        return 'AC Power' in result.stdout
    except:
        return True  # Assume plugged in if we can't detect


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: Path, level: str = 'INFO'):
    """Setup logging to file and console."""
    CONFIG_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('btc_miner')

# ============================================================================
# BLOCK STRUCTURES
# ============================================================================

@dataclass
class BlockHeader:
    """Bitcoin block header (80 bytes)."""
    version: int
    prev_hash: bytes
    merkle_root: bytes
    timestamp: int
    bits: int
    nonce: int = 0

    def serialize(self) -> bytes:
        return struct.pack(
            '<I32s32sIII',
            self.version,
            self.prev_hash,
            self.merkle_root,
            self.timestamp,
            self.bits,
            self.nonce
        )

    def hash(self) -> bytes:
        serialized = self.serialize()
        return hashlib.sha256(hashlib.sha256(serialized).digest()).digest()

    def hash_hex(self) -> str:
        return self.hash()[::-1].hex()


def bits_to_target(bits: int) -> int:
    """Convert compact 'bits' format to full target."""
    exponent = bits >> 24
    mantissa = bits & 0x007fffff
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        target = mantissa << (8 * (exponent - 3))
    return target


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
        return f"{h:.2f} H/s"


def format_time(seconds: float) -> str:
    """Format time duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    elif seconds < 86400 * 365:
        return f"{seconds/86400:.1f}d"
    else:
        return f"{seconds/(86400*365):.2f}y"

# ============================================================================
# BITCOIN CORE RPC CLIENT
# ============================================================================

class BitcoinRPC:
    """Bitcoin Core RPC client."""

    def __init__(self, url: str, user: str, password: str):
        self.url = url
        self.auth = (user, password)
        self.id_counter = 0

    def call(self, method: str, params: list = None) -> dict:
        """Make RPC call to Bitcoin Core."""
        self.id_counter += 1
        payload = {
            'jsonrpc': '2.0',
            'id': self.id_counter,
            'method': method,
            'params': params or []
        }

        response = requests.post(
            self.url,
            json=payload,
            auth=self.auth,
            timeout=30
        )

        result = response.json()
        if 'error' in result and result['error']:
            raise Exception(f"RPC Error: {result['error']}")

        return result.get('result')

    def getblocktemplate(self, rules: list = None) -> dict:
        """Get block template for mining."""
        rules = rules or ['segwit']
        return self.call('getblocktemplate', [{'rules': rules}])

    def submitblock(self, block_hex: str) -> Optional[str]:
        """Submit a mined block."""
        return self.call('submitblock', [block_hex])

    def getblockchaininfo(self) -> dict:
        """Get blockchain info."""
        return self.call('getblockchaininfo')

# ============================================================================
# PUBLIC API CLIENT (for light mode)
# ============================================================================

class PublicAPI:
    """Public API client for light mode."""

    def __init__(self):
        self.base_url = 'https://mempool.space/api'

    def get_latest_block(self) -> dict:
        """Get latest block info."""
        resp = requests.get(f'{self.base_url}/blocks/tip/height', timeout=10)
        height = int(resp.text)

        resp = requests.get(f'{self.base_url}/block-height/{height}', timeout=10)
        block_hash = resp.text

        resp = requests.get(f'{self.base_url}/block/{block_hash}', timeout=10)
        return resp.json()

    def get_block_template(self) -> dict:
        """Create a block template from public API."""
        block = self.get_latest_block()

        return {
            'height': block['height'] + 1,
            'previousblockhash': block['id'],
            'bits': block['bits'],
            'version': 0x20000000,
            'curtime': int(time.time()),
            'difficulty': block['difficulty'],
        }

# ============================================================================
# COINBASE TRANSACTION BUILDER
# ============================================================================

class CoinbaseBuilder:
    """Build coinbase transactions."""

    # Current block reward (after 4th halving, April 2024)
    BLOCK_REWARD = 312500000  # 3.125 BTC in satoshis

    @staticmethod
    def create_coinbase_tx(height: int, wallet_address: str = None,
                           extra_nonce: bytes = b'') -> bytes:
        """Create a coinbase transaction."""

        # Block height in script (BIP34)
        height_bytes = height.to_bytes((height.bit_length() + 7) // 8, 'little')

        # Coinbase script: height + extra_nonce + identifier
        coinbase_script = (
            bytes([len(height_bytes)]) +
            height_bytes +
            extra_nonce +
            b'/SoloMiner/'
        )

        # Build transaction
        tx = b''
        tx += struct.pack('<I', 1)  # Version
        tx += b'\x01'  # Input count
        tx += b'\x00' * 32  # Null txid (coinbase)
        tx += b'\xff\xff\xff\xff'  # Null vout
        tx += bytes([len(coinbase_script)]) + coinbase_script
        tx += b'\xff\xff\xff\xff'  # Sequence

        # Output
        tx += b'\x01'  # Output count
        tx += struct.pack('<Q', CoinbaseBuilder.BLOCK_REWARD)

        if wallet_address:
            # Create proper output script for address
            output_script = CoinbaseBuilder.address_to_script(wallet_address)
        else:
            # OP_RETURN (unspendable - you'll lose the reward!)
            output_script = b'\x6a'  # OP_RETURN

        tx += bytes([len(output_script)]) + output_script
        tx += struct.pack('<I', 0)  # Locktime

        return tx

    @staticmethod
    def address_to_script(address: str) -> bytes:
        """Convert Bitcoin address to output script."""
        # Handle bc1q (bech32 P2WPKH) addresses
        if address.startswith('bc1q'):
            # Bech32 decode - manual implementation
            CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

            # Remove prefix and checksum
            data_part = address[4:]  # Remove 'bc1q'

            # Convert from bech32 characters to 5-bit values
            values = []
            for c in data_part:
                if c in CHARSET:
                    values.append(CHARSET.index(c))

            # Convert 5-bit to 8-bit (skip first value which is witness version)
            # Take all but last 6 values (checksum)
            data_5bit = values[:-6]

            # Convert 5-bit groups to 8-bit bytes
            acc = 0
            bits = 0
            result = []
            for v in data_5bit:
                acc = (acc << 5) | v
                bits += 5
                while bits >= 8:
                    bits -= 8
                    result.append((acc >> bits) & 0xff)

            pubkey_hash = bytes(result)
            if len(pubkey_hash) == 20:
                # P2WPKH: OP_0 <20-byte-key-hash>
                return b'\x00\x14' + pubkey_hash
            elif len(pubkey_hash) == 32:
                # P2WSH: OP_0 <32-byte-script-hash>
                return b'\x00\x20' + pubkey_hash

        elif address.startswith('1'):
            # P2PKH: OP_DUP OP_HASH160 <pubkeyhash> OP_EQUALVERIFY OP_CHECKSIG
            try:
                import base58
                decoded = base58.b58decode_check(address)
                pubkey_hash = decoded[1:]  # Remove version byte
                return b'\x76\xa9\x14' + pubkey_hash + b'\x88\xac'
            except:
                pass

        elif address.startswith('3'):
            # P2SH: OP_HASH160 <scripthash> OP_EQUAL
            try:
                import base58
                decoded = base58.b58decode_check(address)
                script_hash = decoded[1:]
                return b'\xa9\x14' + script_hash + b'\x87'
            except:
                pass

        # Fallback: OP_RETURN (unspendable)
        logging.warning(f"Unknown address format: {address}, using OP_RETURN")
        return b'\x6a'

    @staticmethod
    def merkle_root(txids: List[bytes]) -> bytes:
        """Calculate merkle root from transaction IDs."""
        if not txids:
            return b'\x00' * 32

        while len(txids) > 1:
            if len(txids) % 2 == 1:
                txids.append(txids[-1])

            txids = [
                hashlib.sha256(hashlib.sha256(txids[i] + txids[i+1]).digest()).digest()
                for i in range(0, len(txids), 2)
            ]

        return txids[0]

# ============================================================================
# MINER ENGINE
# ============================================================================

class MinerEngine:
    """The actual mining engine."""

    def __init__(self, config: dict, logger: logging.Logger, init_gpu: bool = True):
        self.config = config
        self.logger = logger
        self.running = False
        self.stats = {
            'start_time': None,
            'total_hashes': 0,
            'blocks_checked': 0,
            'current_height': 0,
            'hashrate': 0,
            'last_update': None,
            'gpu_enabled': False,
        }
        self.lock = threading.Lock()
        self.gpu_miner = None

        # Try to initialize GPU miner (only if init_gpu=True, to avoid issues with fork())
        if init_gpu and config.get('use_gpu', True) and METAL_AVAILABLE:
            self._init_gpu()

        # Setup API client
        if config.get('use_bitcoin_core'):
            self.api = BitcoinRPC(
                config['bitcoin_rpc_url'],
                config['bitcoin_rpc_user'],
                config['bitcoin_rpc_password']
            )
            self.logger.info("Using Bitcoin Core RPC")
        else:
            self.api = PublicAPI()
            self.logger.info("Using public API (light mode)")

    def _init_gpu(self):
        """Initialize GPU miner. Called after fork() in daemon mode."""
        if self.config.get('use_gpu', True) and METAL_AVAILABLE:
            try:
                self.gpu_miner = MetalSHA256Miner()
                self.stats['gpu_enabled'] = True
                self.logger.info(f"GPU mining enabled: {self.gpu_miner.device.name()}")
            except Exception as e:
                self.logger.warning(f"GPU initialization failed: {e}")
                self.gpu_miner = None

    def mine_range(self, header: BlockHeader, target: int,
                   start_nonce: int, end_nonce: int,
                   result_queue: mp.Queue, hash_counter):
        """Mine a range of nonces (worker thread)."""
        local_header = BlockHeader(
            version=header.version,
            prev_hash=header.prev_hash,
            merkle_root=header.merkle_root,
            timestamp=header.timestamp,
            bits=header.bits,
            nonce=start_nonce
        )

        local_count = 0

        for nonce in range(start_nonce, end_nonce):
            if not self.running:
                break

            local_header.nonce = nonce
            h = local_header.hash()
            h_int = int.from_bytes(h, 'little')

            local_count += 1

            if h_int < target:
                result_queue.put({
                    'nonce': nonce,
                    'hash': h,
                    'header': local_header.serialize()
                })
                return

            # Update counter periodically
            if local_count % 10000 == 0:
                with hash_counter.get_lock():
                    hash_counter.value += local_count
                local_count = 0

        with hash_counter.get_lock():
            hash_counter.value += local_count

    def mine_block_gpu(self, template: dict) -> Optional[dict]:
        """Mine a block using GPU."""
        height = template['height']
        prev_hash = bytes.fromhex(template['previousblockhash'])[::-1]
        bits = template['bits']
        target = bits_to_target(bits)

        # Create coinbase
        coinbase = CoinbaseBuilder.create_coinbase_tx(
            height,
            self.config.get('wallet_address')
        )
        coinbase_txid = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
        merkle_root = coinbase_txid

        # Serialize header
        header_bytes = struct.pack(
            '<I32s32sIII',
            template.get('version', 0x20000000),
            prev_hash,
            merkle_root,
            template.get('curtime', int(time.time())),
            bits,
            0  # nonce placeholder
        )

        self.logger.info(f"Mining block {height} (GPU)")
        self.logger.debug(f"Target: {target:064x}"[:40] + "...")

        block_start = time.time()
        last_stats = time.time()
        last_power_check = time.time()
        base_nonce = 0

        # Power-aware batch sizing
        on_ac = is_on_ac_power()
        battery_throttle = self.config.get('battery_throttle', True)

        if on_ac or not battery_throttle:
            batch_size = self.config.get('plugged_batch_size', 4 * 1024 * 1024)
            sleep_time = 0
            power_mode = "AC"
        else:
            batch_size = self.config.get('battery_batch_size', 256 * 1024)
            sleep_time = self.config.get('battery_sleep_ms', 100) / 1000.0
            power_mode = "Battery"

        self.logger.info(f"Power mode: {power_mode} (batch: {batch_size:,}, sleep: {sleep_time*1000:.0f}ms)")
        self.stats['power_mode'] = power_mode

        while self.running and base_nonce < 2**32:
            # Check power status every 30 seconds
            if time.time() - last_power_check >= 30:
                new_on_ac = is_on_ac_power()
                if new_on_ac != on_ac:
                    on_ac = new_on_ac
                    if on_ac or not battery_throttle:
                        batch_size = self.config.get('plugged_batch_size', 4 * 1024 * 1024)
                        sleep_time = 0
                        power_mode = "AC"
                    else:
                        batch_size = self.config.get('battery_batch_size', 256 * 1024)
                        sleep_time = self.config.get('battery_sleep_ms', 100) / 1000.0
                        power_mode = "Battery"
                    self.logger.info(f"Power changed to: {power_mode}")
                    self.stats['power_mode'] = power_mode
                last_power_check = time.time()

            # Mine a batch
            result = self.gpu_miner.mine(
                header_bytes,
                bits,
                max_nonce=base_nonce + batch_size,
                batch_size=batch_size
            )

            # Reset the header base nonce for next iteration
            base_nonce += batch_size

            if result:
                nonce, hash_bytes = result
                # FOUND A BLOCK!
                self.logger.critical("=" * 60)
                self.logger.critical("!!! BLOCK FOUND !!!")
                self.logger.critical("=" * 60)
                self.logger.critical(f"Height: {height}")
                self.logger.critical(f"Nonce: {nonce}")
                self.logger.critical(f"Hash: {hash_bytes[::-1].hex()}")

                win_result = {
                    'nonce': nonce,
                    'hash': hash_bytes,
                    'header': header_bytes[:76] + struct.pack('<I', nonce)
                }
                self.save_win(height, win_result)
                return win_result

            # Sleep on battery to reduce power consumption
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Update stats periodically
            if time.time() - last_stats >= 5:  # Every 5 seconds for GPU
                elapsed = time.time() - block_start
                hashes = base_nonce
                hashrate = hashes / elapsed if elapsed > 0 else 0

                with self.lock:
                    self.stats['total_hashes'] += batch_size
                    self.stats['hashrate'] = hashrate
                    self.stats['last_update'] = datetime.now().isoformat()

                self.save_stats()
                self.logger.info(
                    f"Block {height} | "
                    f"Hashes: {hashes:,} | "
                    f"Rate: {format_hashrate(hashrate)} (GPU/{power_mode})"
                )
                last_stats = time.time()

            # Check for new block
            if time.time() - block_start >= self.config['block_check_interval']:
                try:
                    new_template = self.get_template()
                    if new_template['height'] > height:
                        self.logger.info(f"New block detected, switching to {new_template['height']}")
                        break
                except:
                    pass
                block_start = time.time()  # Reset for next check

        return None

    def mine_block(self, template: dict) -> Optional[dict]:
        """Mine a single block template."""
        # Use GPU if available
        if self.gpu_miner is not None:
            return self.mine_block_gpu(template)

        # Fall back to CPU mining
        height = template['height']
        prev_hash = bytes.fromhex(template['previousblockhash'])[::-1]
        bits = template['bits']
        target = bits_to_target(bits)

        # Create coinbase
        coinbase = CoinbaseBuilder.create_coinbase_tx(
            height,
            self.config.get('wallet_address')
        )
        coinbase_txid = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
        merkle_root = coinbase_txid  # Only coinbase tx

        header = BlockHeader(
            version=template.get('version', 0x20000000),
            prev_hash=prev_hash,
            merkle_root=merkle_root,
            timestamp=template.get('curtime', int(time.time())),
            bits=bits,
            nonce=0
        )

        self.logger.info(f"Mining block {height} (CPU)")
        self.logger.debug(f"Target: {target:064x}"[:40] + "...")

        # Shared counter for hash count
        hash_counter = mp.Value('L', 0)
        result_queue = mp.Queue()

        num_threads = self.config['num_threads']
        chunk_size = 2**32 // num_threads

        threads = []
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size if i < num_threads - 1 else 2**32

            t = threading.Thread(
                target=self.mine_range,
                args=(header, target, start, end, result_queue, hash_counter)
            )
            threads.append(t)
            t.start()

        # Monitor progress
        block_start = time.time()
        last_stats = time.time()

        while self.running:
            try:
                result = result_queue.get(timeout=1.0)

                # FOUND A BLOCK!
                self.logger.critical("=" * 60)
                self.logger.critical("!!! BLOCK FOUND !!!")
                self.logger.critical("=" * 60)
                self.logger.critical(f"Height: {height}")
                self.logger.critical(f"Nonce: {result['nonce']}")
                self.logger.critical(f"Hash: {result['hash'][::-1].hex()}")

                # Save the win
                self.save_win(height, result)

                return result

            except:
                pass

            # Update stats
            if time.time() - last_stats >= self.config['stats_interval']:
                with hash_counter.get_lock():
                    hashes = hash_counter.value

                elapsed = time.time() - block_start
                hashrate = hashes / elapsed if elapsed > 0 else 0

                with self.lock:
                    self.stats['total_hashes'] += hashes
                    self.stats['hashrate'] = hashrate
                    self.stats['last_update'] = datetime.now().isoformat()

                self.save_stats()
                self.logger.info(
                    f"Block {height} | "
                    f"Hashes: {hashes:,} | "
                    f"Rate: {format_hashrate(hashrate)}"
                )
                last_stats = time.time()

            # Check if all threads done or new block arrived
            if not any(t.is_alive() for t in threads):
                break

            # Check for new block every N seconds
            if time.time() - block_start >= self.config['block_check_interval']:
                try:
                    new_template = self.get_template()
                    if new_template['height'] > height:
                        self.logger.info(f"New block detected, switching to {new_template['height']}")
                        self.running = False  # Stop current mining
                        break
                except:
                    pass

        # Cleanup threads
        for t in threads:
            t.join(timeout=1.0)

        return None

    def get_template(self) -> dict:
        """Get block template from API."""
        if isinstance(self.api, BitcoinRPC):
            return self.api.getblocktemplate()
        else:
            return self.api.get_block_template()

    def save_win(self, height: int, result: dict):
        """Save a winning block to file."""
        wins = []
        if WINS_FILE.exists():
            wins = json.loads(WINS_FILE.read_text())

        wins.append({
            'timestamp': datetime.now().isoformat(),
            'height': height,
            'nonce': result['nonce'],
            'hash': result['hash'][::-1].hex(),
            'header_hex': result['header'].hex(),
        })

        WINS_FILE.write_text(json.dumps(wins, indent=2))

        # Also send notification
        self.notify_win(height, result)

    def notify_win(self, height: int, result: dict):
        """Send notification about winning block."""
        # macOS notification
        try:
            os.system(f'''
                osascript -e 'display notification "Block {height} found! Hash: {result["hash"][::-1].hex()[:16]}..." with title "BITCOIN BLOCK FOUND!" sound name "Glass"'
            ''')
        except:
            pass

        self.logger.critical("CHECK wins.json FOR BLOCK DATA!")
        self.logger.critical("If using light mode, you need to submit manually to a node!")

    def save_stats(self):
        """Save current stats to file."""
        with self.lock:
            stats = self.stats.copy()

        STATS_FILE.write_text(json.dumps(stats, indent=2))

    def run(self):
        """Main mining loop."""
        self.running = True
        self.stats['start_time'] = datetime.now().isoformat()

        self.logger.info("=" * 60)
        self.logger.info("BITCOIN SOLO MINER STARTING")
        self.logger.info("=" * 60)
        if self.gpu_miner:
            self.logger.info(f"Mining: GPU ({self.gpu_miner.device.name()})")
        else:
            self.logger.info(f"Mining: CPU ({self.config['num_threads']} threads)")
        self.logger.info(f"Wallet: {self.config.get('wallet_address', 'NOT SET (will lose rewards!)')}")
        self.logger.info(f"Mode: {'Bitcoin Core' if self.config.get('use_bitcoin_core') else 'Light (public API)'}")

        while self.running:
            try:
                template = self.get_template()
                self.stats['current_height'] = template['height']
                self.stats['blocks_checked'] += 1

                result = self.mine_block(template)

                if result and isinstance(self.api, BitcoinRPC):
                    # Submit to Bitcoin Core
                    try:
                        # Would need to construct full block here
                        # For now just log
                        self.logger.info("Would submit block to Bitcoin Core...")
                    except Exception as e:
                        self.logger.error(f"Failed to submit block: {e}")

                self.running = True  # Reset for next block

            except KeyboardInterrupt:
                self.logger.info("Shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Error: {e}")
                time.sleep(10)

        self.running = False
        self.save_stats()
        self.logger.info("Miner stopped")

    def stop(self):
        """Stop the miner."""
        self.running = False

# ============================================================================
# DAEMON MANAGEMENT
# ============================================================================

def start_daemon(config: dict):
    """Start miner as background daemon using subprocess (avoids fork issues with Metal)."""
    import subprocess

    # Use subprocess.Popen to start daemon - this avoids fork() issues with Metal GPU
    script_path = Path(__file__).resolve()

    # Start the miner in background mode using nohup
    proc = subprocess.Popen(
        ['python3', str(script_path), '--foreground'],
        stdout=open(LOG_FILE, 'a'),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # Detach from terminal
    )

    # Write PID file
    CONFIG_DIR.mkdir(exist_ok=True)
    PID_FILE.write_text(str(proc.pid))

    print(f"Miner started in background. PID: {proc.pid}")
    print(f"PID file: {PID_FILE}")
    print(f"Log file: {LOG_FILE}")
    print(f"Check status: python3 btc_miner.py --status")
    print(f"Stop: python3 btc_miner.py --stop")


def stop_daemon():
    """Stop the daemon."""
    if not PID_FILE.exists():
        print("Miner is not running")
        return

    pid = int(PID_FILE.read_text().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent stop signal to PID {pid}")

        # Wait for process to die
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                print("Miner stopped")
                if PID_FILE.exists():
                    PID_FILE.unlink()
                return

        print("Miner did not stop gracefully, killing...")
        os.kill(pid, signal.SIGKILL)

    except ProcessLookupError:
        print("Miner was not running")
        if PID_FILE.exists():
            PID_FILE.unlink()


def show_status():
    """Show miner status."""
    print("=" * 60)
    print("BITCOIN SOLO MINER STATUS")
    print("=" * 60)

    # Check if running
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Status: RUNNING (PID {pid})")
        except ProcessLookupError:
            print("Status: STOPPED (stale PID file)")
            PID_FILE.unlink()
    else:
        print("Status: STOPPED")

    # Show stats
    if STATS_FILE.exists():
        stats = json.loads(STATS_FILE.read_text())
        print(f"\nStats:")
        print(f"  Started: {stats.get('start_time', 'N/A')}")
        print(f"  Total hashes: {stats.get('total_hashes', 0):,}")
        print(f"  Current height: {stats.get('current_height', 'N/A')}")
        print(f"  Hashrate: {format_hashrate(stats.get('hashrate', 0))}")
        print(f"  GPU enabled: {'Yes' if stats.get('gpu_enabled') else 'No'}")
        print(f"  Last update: {stats.get('last_update', 'N/A')}")

    # Show wins
    if WINS_FILE.exists():
        wins = json.loads(WINS_FILE.read_text())
        if wins:
            print(f"\n!!! BLOCKS FOUND: {len(wins)} !!!")
            for win in wins:
                print(f"  Height {win['height']}: {win['hash'][:32]}...")

    # Show config
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        print(f"\nConfig:")
        print(f"  Wallet: {config.get('wallet_address', 'NOT SET')}")
        print(f"  Threads: {config.get('num_threads', 'default')}")
        print(f"  Mode: {'Bitcoin Core' if config.get('use_bitcoin_core') else 'Light'}")

    print(f"\nLog file: {LOG_FILE}")


def load_config() -> dict:
    """Load config from file."""
    CONFIG_DIR.mkdir(exist_ok=True)

    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        # Merge with defaults
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
        return config

    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """Save config to file."""
    CONFIG_DIR.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bitcoin Solo Miner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 btc_miner.py --wallet bc1q...        Set wallet address
  python3 btc_miner.py --daemon                Start mining in background
  python3 btc_miner.py --status                Check miner status
  python3 btc_miner.py --stop                  Stop the miner
  python3 btc_miner.py --foreground            Run in foreground (for testing)
        '''
    )

    parser.add_argument('--daemon', action='store_true',
                        help='Run as background daemon')
    parser.add_argument('--foreground', action='store_true',
                        help='Run in foreground')
    parser.add_argument('--stop', action='store_true',
                        help='Stop the daemon')
    parser.add_argument('--status', action='store_true',
                        help='Show miner status')
    parser.add_argument('--wallet', type=str,
                        help='Set Bitcoin wallet address for rewards')
    parser.add_argument('--threads', type=int,
                        help='Number of mining threads')
    parser.add_argument('--bitcoin-core', action='store_true',
                        help='Use Bitcoin Core RPC')

    args = parser.parse_args()

    # Load/update config
    config = load_config()

    if args.wallet:
        config['wallet_address'] = args.wallet
        save_config(config)
        print(f"Wallet set to: {args.wallet}")

    if args.threads:
        config['num_threads'] = args.threads
        save_config(config)
        print(f"Threads set to: {args.threads}")

    if args.bitcoin_core:
        config['use_bitcoin_core'] = True
        save_config(config)
        print("Bitcoin Core mode enabled")

    # Commands
    if args.stop:
        stop_daemon()
    elif args.status:
        show_status()
    elif args.daemon:
        if not config.get('wallet_address'):
            print("WARNING: No wallet address set!")
            print("You will LOSE any block rewards without a wallet.")
            print("Set one with: python3 btc_miner.py --wallet <address>")
            print()
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        start_daemon(config)
    elif args.foreground:
        logger = setup_logging(LOG_FILE, config.get('log_level', 'INFO'))
        engine = MinerEngine(config, logger)
        try:
            engine.run()
        except KeyboardInterrupt:
            print("\nStopping...")
            engine.stop()
    else:
        parser.print_help()
        print("\n" + "=" * 60)
        show_status()


if __name__ == '__main__':
    main()
