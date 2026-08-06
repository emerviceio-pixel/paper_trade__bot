import os
import sys
import asyncio
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from decimal import Decimal

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from pybit.unified_trading import WebSocket
import numpy as np
from loguru import logger

logger.remove()  # Remove default handler
logger.add(
    "logs/trading.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO"
)
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)

# Load environment variables
env_path = Path(__file__).parent.parent / 'config' / '.env'
load_dotenv(env_path)

# --- CONFIGURATION ---
SYMBOL = os.getenv('SYMBOL', '1000PEPEUSDT')
LEVERAGE = int(os.getenv('LEVERAGE', 5))
MAX_TRADE_SIZE = float(os.getenv('MAX_TRADE_SIZE', 10.0))
MIN_BALANCE = float(os.getenv('MIN_BALANCE', 9.50))
LOSS_STREAK_LIMIT = int(os.getenv('LOSS_STREAK_LIMIT', 2))
MAX_SPREAD = float(os.getenv('MAX_SPREAD', 0.0015))

TARGET_PERCENT = 0.0015  # 0.15%
STOP_PERCENT = 0.0012    # 0.12%


# --- WEBSOCKET BOT ---
class WebSocketTradingBot:
    def __init__(self, loop=None):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("❌ API keys not found in .env file!")
        
        # Capture the primary async event loop running on main thread
        self.loop = loop or asyncio.get_event_loop()
        
        # HTTP client for placing orders
        self.session = HTTP(
            testnet=False,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        
        self.ws = None  # Explicit declaration
        
        # Real-time data storage
        self.current_price = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.last_update_time = 0
        
        # Cached balance to eliminate heavy blocking REST API operations
        self.cached_balance = 0.0
        self.last_balance_check = 0
        
        # Price buffer for indicators (last 100 ticks)
        self.price_buffer = []
        self.bid_buffer = []
        self.ask_buffer = []
        
        # Trade tracking
        self.loss_streak = 0
        self.cooldown_until = 0  # Non-blocking timestamp tracking
        self.is_trading = False
        self.last_trade_time = 0
        
        # Threading for concurrent execution
        self.websocket_thread = None
        self.running = False
        
        logger.info(f"✅ WebSocket bot initialized for {SYMBOL}")

    # ============================================
    # WEBSOCKET DATA HANDLERS (Thread-Safe)
    # ============================================
    
    def handle_orderbook_data(self, message):
        """Handle real-time order book updates safely from background thread"""
        try:
            data = message.get('data', {})
            if not data:
                return
            
            # Extract bids and asks (Handle delta updates where one side might be omitted)
            bids = data.get('b', [])
            asks = data.get('a', [])
            
            if bids:
                self.best_bid = float(bids[0][0])
            if asks:
                self.best_ask = float(asks[0][0])
                
            if self.best_bid > 0 and self.best_ask > 0:
                self.current_price = (self.best_bid + self.best_ask) / 2
                self.last_update_time = time.time()
                
                # Store structural ticks into memory buffers
                self.price_buffer.append(self.current_price)
                self.bid_buffer.append(self.best_bid)
                self.ask_buffer.append(self.best_ask)
                
                if len(self.price_buffer) > 100:
                    self.price_buffer.pop(0)
                    self.bid_buffer.pop(0)
                    self.ask_buffer.pop(0)

                if len(self.price_buffer) % 10 == 0:  # Save every 10th update
                    self.save_price_data()

                # Thread-safely ship the function call to the primary thread loop context
                asyncio.run_coroutine_threadsafe(self.on_price_update(), self.loop)
                
        except Exception as e:
            logger.error(f"Order book handler error: {e}")

    def handle_trade_data(self, message):
        """Handle real-time trade updates"""
        try:
            data = message.get('data', {})
            if not data:
                return

            if isinstance(data, list) and data:
                first_entry = data[0]
                if isinstance(first_entry, dict):
                    data = first_entry
                elif isinstance(first_entry, (list, tuple)) and len(first_entry) >= 2:
                    data = {
                        'p': first_entry[0],
                        'v': first_entry[1],
                    }

            if not isinstance(data, dict):
                logger.debug(f"Unsupported trade payload shape: {type(data).__name__}")
                return

            trade_price = float(data.get('p', 0))
            trade_volume = float(data.get('v', 0))
            logger.debug(f"📊 Trade: {trade_volume} {SYMBOL} @ {trade_price}")

        except Exception as e:
            logger.error(f"Trade handler error: {e}")

    # ============================================
    # ASYNC UTILITIES (Non-blocking network wraps)
    # ============================================
    
    async def async_get_balance(self):
        """Fetch wallet balance asynchronously using an executor pool"""
        now = time.time()
        # Throttle live balance requests to once every 10 seconds max
        if now - self.last_balance_check < 10 and self.cached_balance > 0:
            return self.cached_balance
            
        try:
            response = await self.loop.run_in_executor(
                None, 
                lambda: self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            )
            # Safe layout extraction of totalEquity parameter
            balance_str = response.get('result', {}).get('list', [{}])[0].get('totalEquity', '0')
            self.cached_balance = float(balance_str)
            self.last_balance_check = now
            return self.cached_balance
        except Exception as e:
            logger.error(f"Failed to fetch balance asynchronously: {e}")
            return self.cached_balance if self.cached_balance > 0 else 0.0

    def calculate_rsi(self, period=14):
        """Calculate RSI from price buffer cleanly"""
        if len(self.price_buffer) < period + 1:
            return 50.0
        
        gains = 0
        losses = 0
        
        for i in range(1, period + 1):
            diff = self.price_buffer[-i] - self.price_buffer[-i-1]
            if diff > 0:
                gains += diff
            else:
                losses += abs(diff)
        
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100.0 - (100.0 / (1 + rs))

    async def execute_trade(self, side, quantity, tp_price, sl_price):
        """Execute trade using atomic TP/SL parameters via a thread executor"""
        try:
            # 1. Asynchronously wipe old resting tracking orders first
            await self.loop.run_in_executor(
                None,
                lambda: self.session.cancel_all_orders(category="linear", symbol=SYMBOL)
            )
            
            # 2. Build the order execution statement
            # Attaching takeProfit/stopLoss parameter fields directly executes a secure atomic bracket order
            order_args = {
                "category": "linear",
                "symbol": SYMBOL,
                "side": side,
                "orderType": "Market",
                "qty": str(quantity),
                "timeInForce": "GTC",
                "takeProfit": str(tp_price),
                "stopLoss": str(sl_price),
                "tpOrderType": "Market",
                "slOrderType": "Market"
            }
            
            entry = await self.loop.run_in_executor(
                None,
                lambda: self.session.place_order(**order_args)
            )
            
            if not entry or 'result' not in entry:
                return False
            
            logger.info(f"✅ Market Bracket Entry Confirmed! ID: {entry['result'].get('orderId')}")
            logger.info(f"   🎯 Attached TP: {tp_price} | 🛑 Attached SL: {sl_price}")
            
            self.loss_streak = 0  # Reset streak counter loop state
            return True
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False

    # ============================================
    # TRADING LOGIC STRATEGY RUNNER
    # ============================================
    
    async def on_price_update(self):
        """Execute trading logic safely inside the primary loop thread context"""
        # 1. Immediately drop out if the bot has been flagged to stop
        if not self.running:
            return

        if self.is_trading:
            return
        
        now = time.time()
        if now - self.last_trade_time < 0.1:
            return
            
        if now < self.cooldown_until:
            return
        
        if len(self.price_buffer) < 50:
            return
        
        try:
            self.is_trading = True
            
            if self.current_price <= 0 or self.best_bid <= 0 or self.best_ask <= 0:
                self.is_trading = False
                return
            
            spread = (self.best_ask - self.best_bid) / self.best_bid
            if spread > MAX_SPREAD:
                self.is_trading = False
                return
            
            # Fetch balance via non-blocking async wrapper
            balance = await self.async_get_balance()
            if balance < MIN_BALANCE:
                # ✅ FIXED: Kill the system state switches BEFORE logging
                self.running = False  # Blocks any future incoming websocket updates
                
                logger.critical(f"💀 Balance threshold breached! Account Equity: ${balance:.2f} < Minimum: ${MIN_BALANCE:.2f}. Halting execution system.")
                
                if self.ws:
                    self.ws.exit()  # Clean up and slam down the background websocket sockets
                return
            
            if self.loss_streak >= LOSS_STREAK_LIMIT:
                logger.warning(f"⛔ {self.loss_streak} losses. Cooling down for 10 minutes.")
                self.cooldown_until = now + 600
                self.loss_streak = 0
                self.is_trading = False
                return
            
            # --- INDICATORS ---
            rsi = self.calculate_rsi()
            volatility = np.std(self.price_buffer[-20:])
            sma = np.mean(self.price_buffer[-50:])
            price_vs_sma = (self.current_price - sma) / sma
            
            side = None
            entry_price = self.current_price
            
            if rsi > 65 and price_vs_sma > 0.002:
                side = "Sell"
                tp_price = round(entry_price * (1 - TARGET_PERCENT), 8)
                sl_price = round(entry_price * (1 + STOP_PERCENT), 8)
            elif rsi < 35 and price_vs_sma < -0.002:
                side = "Buy"
                tp_price = round(entry_price * (1 + TARGET_PERCENT), 8)
                sl_price = round(entry_price * (1 - STOP_PERCENT), 8)
            else:
                self.is_trading = False
                return

            if not side:
                self.is_trading = False
                return

            logger.info(f"🎯 {side} signal detected!")
            logger.info(f"  Price: {entry_price} | RSI: {rsi:.2f} | Volatility: {volatility:.8f}")
            
            # Formulate asset lot quantity parameters cleanly
            quantity = (MAX_TRADE_SIZE * LEVERAGE) / entry_price
            quantity = float(Decimal(str(quantity)).quantize(Decimal('1')))
            
            if quantity == 0:
                logger.warning("⚠️ Calculated trade quantity returned 0 lot sizing.")
                self.is_trading = False
                return
            
            # Route execution out via thread safe await channel
            success = await self.execute_trade(side, quantity, tp_price, sl_price)
            
            if success:
                self.last_trade_time = time.time()
                logger.success(f"✅ Trade executed!")
            else:
                logger.error(f"❌ Trade failed!")
                self.loss_streak += 1
                
            self.is_trading = False
            
        except Exception as e:
            logger.error(f"Trading logic execution error loop crash: {e}")
            self.is_trading = False

    # ============================================
    # PRICE DATA SAVING (ADD THIS INSIDE THE CLASS)
    # ============================================
    
    def save_price_data(self):
        """Save price data to CSV for backtesting"""
        import csv
        from pathlib import Path
        
        csv_file = Path(__file__).parent.parent / 'data' / 'price_history.csv'
        csv_file.parent.mkdir(exist_ok=True)
        self.last_saved_price = 0
        
        file_exists = csv_file.exists()
        
        # Calculate total volume from top 5 levels
        bid_volume = sum(self.bid_buffer[:5]) if len(self.bid_buffer) >= 5 else 0
        ask_volume = sum(self.ask_buffer[:5]) if len(self.ask_buffer) >= 5 else 0
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'price', 'bid', 'ask', 'spread',
                    'bid_volume', 'ask_volume', 'imbalance'  # New columns
                ])
            
            spread = (self.best_ask - self.best_bid) / self.best_bid if self.best_bid > 0 else 0
            imbalance = bid_volume / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.5

            if abs(self.current_price - self.last_saved_price) < 0.000001:  # Skip tiny changes
                return
            self.last_saved_price = self.current_price
            
            writer.writerow([
                datetime.utcnow().isoformat(),
                self.current_price,
                self.best_bid,
                self.best_ask,
                spread,
                bid_volume,
                ask_volume,
                imbalance
            ])       

    # ============================================
    # WEBSOCKET CONNECTION MANAGEMENT
    # ============================================
    
    def start_websocket(self):
        """Start WebSocket connection in a separate thread"""
        try:
            logger.info("🔌 Connecting to Bybit WebSocket...")
            # Initialize WebSocket client
            self.ws = WebSocket(
                testnet=False,
                channel_type='linear',
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
            # Fixed parameter signature rules (Positional depth level mapping)
            self.ws.orderbook_stream(
                50,
                symbol=SYMBOL,
                callback=self.handle_orderbook_data
            )
            self.ws.trade_stream(
                symbol=SYMBOL,
                callback=self.handle_trade_data
            )
            logger.info("✅ WebSocket connected and streaming!")
        except Exception as e:
            logger.error(f"WebSocket error initialization failed: {e}")
            self.running = False

    async def run(self):
        """Main bot loop managed via Asyncio"""
        logger.info(f"🚀 Starting WebSocket bot on {SYMBOL}")
        logger.info(f"   Leverage: {LEVERAGE}x")
        logger.info(f"   Trading hours: 8:00 - 17:00 UTC")
        
        self.running = True
        
        # Start WebSocket background framework thread
        self.websocket_thread = threading.Thread(target=self.start_websocket, daemon=True)
        self.websocket_thread.start()
        
        # Keep main runtime alive and safely monitor state
        while self.running:
            await asyncio.sleep(1)
            current_time = int(time.time())
            if current_time % 60 == 0:  # Monitor balance once a minute
                #Double check state flag before making duplicate REST queries
                if not self.running:
                    break
                    
                balance = await self.async_get_balance()
                logger.info(f"💰 Account Equity Balance: ${balance:.2f}")

                if balance < MIN_BALANCE:
                    logger.critical(f"💀 Supervisor detected low balance (${balance:.2f}). Stopping system runtime.")
                    self.running = False
                    if self.ws:
                        self.ws.exit()
                    break


# ============================================
# ENTRY FILE INTEGRATION
# ============================================

async def main():
    # Pass the active loop reference down into the constructor block
    bot = WebSocketTradingBot(loop=asyncio.get_running_loop())
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user manually.")
    except Exception as e:
        logger.error(f"❌ Fatal error in loop initialization payload: {e}")