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
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("❌ API keys not found in .env file!")
        
        # HTTP client for placing orders
        self.session = HTTP(
            testnet=False,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        
        # Real-time data storage
        self.current_price = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.last_update_time = 0
        
        # Price buffer for indicators (last 100 ticks)
        self.price_buffer = []
        self.bid_buffer = []
        self.ask_buffer = []
        
        # Trade tracking
        self.loss_streak = 0
        self.is_trading = False
        self.last_trade_time = 0
        
        # Threading for concurrent execution
        self.websocket_thread = None
        self.running = False
        
        logger.info(f"✅ WebSocket bot initialized for {SYMBOL}")

    # ============================================
    # WEBSOCKET DATA HANDLER
    # ============================================
    
    def handle_orderbook_data(self, message):
        """Handle real-time order book updates"""
        try:
            # Parse WebSocket message
            data = message.get('data', {})
            if not data:
                return
            
            # Extract bids and asks
            bids = data.get('b', [])
            asks = data.get('a', [])
            
            if bids and asks:
                self.best_bid = float(bids[0][0])
                self.best_ask = float(asks[0][0])
                self.current_price = (self.best_bid + self.best_ask) / 2
                self.last_update_time = time.time()
                
                # Store in buffers
                self.price_buffer.append(self.current_price)
                self.bid_buffer.append(self.best_bid)
                self.ask_buffer.append(self.best_ask)
                
                # Keep buffers small (last 100 ticks)
                if len(self.price_buffer) > 100:
                    self.price_buffer.pop(0)
                    self.bid_buffer.pop(0)
                    self.ask_buffer.pop(0)
                
                # Trigger trading logic on each update
                asyncio.create_task(self.on_price_update())
                
        except Exception as e:
            logger.error(f"Order book handler error: {e}")

    def handle_trade_data(self, message):
        """Handle real-time trade updates"""
        try:
            data = message.get('data', {})
            if not data:
                return
            
            # Log trades for analysis
            trade_price = float(data.get('p', 0))
            trade_volume = float(data.get('v', 0))
            
            logger.debug(f"📊 Trade: {trade_volume} {SYMBOL} @ {trade_price}")
            
        except Exception as e:
            logger.error(f"Trade handler error: {e}")

    # ============================================
    # TRADING LOGIC (Runs on every WebSocket update)
    # ============================================
    
    async def on_price_update(self):
        """Execute trading logic on every price update"""
        
        # Avoid multiple concurrent trades
        if self.is_trading:
            return
        
        # Minimum time between trades (100ms)
        if time.time() - self.last_trade_time < 0.1:
            return
        
        # Need minimum data for indicators
        if len(self.price_buffer) < 20:
            return
        
        try:
            self.is_trading = True
            
            # --- SAFETY CHECKS ---
            
            # 1. Check if price is valid
            if self.current_price <= 0:
                self.is_trading = False
                return
            
            # 2. Check spread
            if self.best_bid <= 0 or self.best_ask <= 0:
                self.is_trading = False
                return
            
            spread = (self.best_ask - self.best_bid) / self.best_bid
            if spread > MAX_SPREAD:
                self.is_trading = False
                return
            
            # 3. Check balance
            balance = self.get_balance()
            if balance < MIN_BALANCE:
                logger.critical(f"💀 Balance {balance} < ${MIN_BALANCE}")
                self.is_trading = False
                return
            
            # 4. Check loss streak
            if self.loss_streak >= LOSS_STREAK_LIMIT:
                logger.warning(f"⛔ {self.loss_streak} losses. Cooling down.")
                await asyncio.sleep(600)  # 10 min cooldown
                self.loss_streak = 0
                self.is_trading = False
                return
            
            # 5. Check imbalance (using real-time bid/ask volumes from WebSocket)
            # For simplicity, use the order book imbalance from the update
            
            # --- DETERMINE DIRECTION ---
            # Calculate RSI from real-time buffer
            rsi = self.calculate_rsi()
            
            # Calculate volatility
            volatility = np.std(self.price_buffer[-20:]) if len(self.price_buffer) >= 20 else 0
            
            # Calculate price vs SMA
            sma = np.mean(self.price_buffer[-50:]) if len(self.price_buffer) >= 50 else self.current_price
            price_vs_sma = (self.current_price - sma) / sma
            
            # Trading decision
            side = None
            entry_price = self.current_price
            
            # Look for momentum
            if rsi > 65 and price_vs_sma > 0.002:
                side = "Sell"  # Overbought, short
                tp_price = round(entry_price * (1 - TARGET_PERCENT), 8)
                sl_price = round(entry_price * (1 + STOP_PERCENT), 8)
                close_side = "Buy"
                
            elif rsi < 35 and price_vs_sma < -0.002:
                side = "Buy"   # Oversold, long
                tp_price = round(entry_price * (1 + TARGET_PERCENT), 8)
                sl_price = round(entry_price * (1 - STOP_PERCENT), 8)
                close_side = "Sell"
                
            else:
                # Stable market - use order book imbalance
                # This would require more sophisticated order book analysis
                self.is_trading = False
                return
            
            if not side:
                self.is_trading = False
                return
            
            # --- EXECUTE TRADE ---
            logger.info(f"🎯 {side} signal detected!")
            logger.info(f"  Price: {entry_price}")
            logger.info(f"  RSI: {rsi:.2f}")
            logger.info(f"  Volatility: {volatility:.8f}")
            logger.info(f"  Price vs SMA: {price_vs_sma:.4%}")
            
            # Calculate quantity
            quantity = (MAX_TRADE_SIZE * LEVERAGE) / entry_price
            quantity = round(quantity, 0)
            
            if quantity == 0:
                self.is_trading = False
                return
            
            # Place order
            success = await self.execute_trade(side, quantity, tp_price, sl_price)
            
            if success:
                self.last_trade_time = time.time()
                logger.success(f"✅ Trade executed!")
            else:
                logger.error(f"❌ Trade failed!")
                self.loss_streak += 1
            
            self.is_trading = False
            
        except Exception as e:
            logger.error(f"Trading logic error: {e}")
            self.is_trading = False

    def calculate_rsi(self, period=14):
        """Calculate RSI from price buffer"""
        if len(self.price_buffer) < period + 1:
            return 50
        
        gains = 0
        losses = 0
        
        for i in range(1, period + 1):
            diff = self.price_buffer[-i] - self.price_buffer[-i-1]
            if diff > 0:
                gains += diff
            else:
                losses += abs(diff)
        
        if losses == 0:
            return 100
        rs = gains / losses
        return 100 - (100 / (1 + rs))

    def get_balance(self):
        """Get wallet balance"""
        try:
            account = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            balance = float(account['result']['list'][0]['totalEquity'])
            return balance
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return 0.0

    async def execute_trade(self, side, quantity, tp_price, sl_price):
        """Execute trade with TP and SL"""
        try:
            # Cancel old orders first
            self.session.cancel_all_orders(
                category="linear",
                symbol=SYMBOL
            )
            
            # 1. Market entry
            entry = self.session.place_order(
                category="linear",
                symbol=SYMBOL,
                side=side,
                orderType="Market",
                qty=str(quantity),
                timeInForce="GTC"
            )
            
            if not entry or 'result' not in entry:
                return False
            
            logger.info(f"✅ Entry: {entry['result']['orderId']}")
            
            # 2. Take Profit (Limit)
            tp = self.session.place_order(
                category="linear",
                symbol=SYMBOL,
                side="Sell" if side == "Buy" else "Buy",
                orderType="Limit",
                qty=str(quantity),
                price=str(tp_price),
                timeInForce="GTC"
            )
            
            # 3. Stop Loss (Market)
            sl = self.session.place_order(
                category="linear",
                symbol=SYMBOL,
                side="Sell" if side == "Buy" else "Buy",
                orderType="Market",
                qty=str(quantity),
                timeInForce="GTC",
                triggerPrice=str(sl_price),
                triggerDirection=1 if side == "Sell" else 2
            )
            
            logger.info(f"🎯 TP: {tp_price} | 🛑 SL: {sl_price}")
            
            # Reset loss streak on successful trade
            self.loss_streak = 0
            return True
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False

    # ============================================
    # WEBSOCKET CONNECTION MANAGEMENT
    # ============================================
    
    def start_websocket(self):
        """Start WebSocket connection in a separate thread"""
        try:
            logger.info("🔌 Connecting to Bybit WebSocket...")
            
            # Order book stream
            ws = WebSocket(
                testnet=False,
                channel_type="linear",
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
            
            # Subscribe to order book
            ws.orderbook_stream(
                symbol=SYMBOL,
                callback=self.handle_orderbook_data,
                depth=10
            )
            
            # Subscribe to trades
            ws.trade_stream(
                symbol=SYMBOL,
                callback=self.handle_trade_data
            )
            
            logger.info("✅ WebSocket connected and streaming!")
            
            # Keep running
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.running = False

    async def run(self):
        """Main bot loop"""
        logger.info(f"🚀 Starting WebSocket bot on {SYMBOL}")
        logger.info(f"   Leverage: {LEVERAGE}x")
        logger.info(f"   Trading hours: 8:00 - 17:00 UTC")
        
        # Start WebSocket in background thread
        self.running = True
        self.websocket_thread = threading.Thread(
            target=self.start_websocket,
            daemon=True
        )
        self.websocket_thread.start()
        
        # Keep main thread alive
        while self.running:
            await asyncio.sleep(1)
            
            # Periodic balance check
            if int(time.time()) % 60 == 0:  # Every minute
                balance = self.get_balance()
                logger.info(f"💰 Balance: ${balance:.2f}")
                
                if balance < MIN_BALANCE:
                    logger.critical(f"💀 Balance below ${MIN_BALANCE}. Stopping.")
                    self.running = False
                    break

# ============================================
# MAIN
# ============================================

async def main():
    bot = WebSocketTradingBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user.")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")