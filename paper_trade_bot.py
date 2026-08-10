import os
import sys
import asyncio
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from decimal import Decimal
from collections import deque

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from pybit.unified_trading import WebSocket
import numpy as np
from loguru import logger

logger.remove()  # Remove default handler
logger.add(
    "logs/paper_trading.log",
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
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# --- CONFIGURATION ---
SYMBOL = os.getenv('SYMBOL', '1000PEPEUSDT')
LEVERAGE = int(os.getenv('LEVERAGE', 5))
MAX_TRADE_SIZE = float(os.getenv('MAX_TRADE_SIZE', 10.0))
MIN_BALANCE = float(os.getenv('MIN_BALANCE', 9.50))
LOSS_STREAK_LIMIT = int(os.getenv('LOSS_STREAK_LIMIT', 2))
MAX_SPREAD = float(os.getenv('MAX_SPREAD', 0.0015))

TARGET_PERCENT = 0.003  # 0.3%
STOP_PERCENT = 0.002    # 0.2%

# --- PAPER TRADING CONFIGURATION ---
PAPER_INITIAL_BALANCE = float(os.getenv('PAPER_INITIAL_BALANCE', 10.0))
PAPER_TRADE_LOG = Path(__file__).parent / 'data' / 'paper_trades.json'
PAPER_PNL_LOG = Path(__file__).parent / 'data' / 'paper_pnl.csv'

# --- PAPER TRADING STATE ----
class PaperTradingState:
    """Maintains paper trading state (position, P&L, history)"""
    def __init__(self, initial_balance=PAPER_INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0  # Current position size (positive = long, negative = short)
        self.entry_price = 0.0
        self.current_price = 0.0
        self.trades = []  # List of completed trades
        self.open_trade = None  # Current open trade info
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
    def open_position(self, side, quantity, entry_price, tp_price, sl_price):
        """Open a paper position"""
        if self.position != 0:
            logger.warning("⚠️ Position already open, closing first")
            self.close_position(entry_price)  # Close at current price
        
        self.position = quantity if side == "Buy" else -quantity
        self.entry_price = entry_price
        self.current_price = entry_price
        
        self.open_trade = {
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'open_time': datetime.utcnow().isoformat(),
            'status': 'open'
        }
        
        logger.info(f"📊 [PAPER] Opened {side} position: {quantity:.4f} @ {entry_price:.8f}")
        logger.info(f"   🎯 TP: {tp_price:.8f} | 🛑 SL: {sl_price:.8f}")
        
    def close_position(self, exit_price):
        """Close paper position and calculate P&L"""
        if self.position == 0 or not self.open_trade:
            return 0.0
        
        # Calculate P&L
        if self.position > 0:  # Long position
            pnl = (exit_price - self.entry_price) * self.position
        else:  # Short position
            pnl = (self.entry_price - exit_price) * abs(self.position)
        
        # Apply leverage to P&L
        pnl = pnl * LEVERAGE

        # Deduct Bybit taker fees (Entry + Exit)
        FEE_RATE = 0.00055  # Bybit taker fee
        position_value = abs(self.position) * self.entry_price
        fees = position_value * FEE_RATE * 2 * LEVERAGE  # Entry + Exit
        pnl = pnl - fees
        
        # Update balance
        self.balance += pnl
        self.total_pnl += pnl
        
        # Update win/loss stats
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Record trade
        trade_record = {
            **self.open_trade,
            'exit_price': exit_price,
            'exit_time': datetime.utcnow().isoformat(),
            'pnl': pnl,
            'pnl_percent': (pnl / (self.entry_price * abs(self.position))) * 100 if self.entry_price > 0 else 0,
            'balance_after': self.balance,
            'status': 'closed'
        }
        self.trades.append(trade_record)
        
        logger.info(f"📊 [PAPER] Closed position @ {exit_price:.8f}")
        logger.info(f"   P&L: ${pnl:.2f} | Balance: ${self.balance:.2f}")
        
        # Reset position
        self.position = 0.0
        self.entry_price = 0.0
        self.open_trade = None
        
        # Save trade to log
        self.save_trade(trade_record)
        
        return pnl
    
    def update_price(self, current_price):
        """Update current price and check TP/SL"""
        self.current_price = current_price
        
        if self.position == 0 or not self.open_trade:
            return
        
        # Check TP/SL
        tp_price = self.open_trade['tp_price']
        sl_price = self.open_trade['sl_price']
        
        if self.position > 0:  # Long position
            if current_price >= tp_price:
                logger.info(f"🎯 [PAPER] Take Profit hit! Closing long position")
                self.close_position(tp_price)
            elif current_price <= sl_price:
                logger.info(f"🛑 [PAPER] Stop Loss hit! Closing long position")
                self.close_position(sl_price)
        else:  # Short position
            if current_price <= tp_price:
                logger.info(f"🎯 [PAPER] Take Profit hit! Closing short position")
                self.close_position(tp_price)
            elif current_price >= sl_price:
                logger.info(f"🛑 [PAPER] Stop Loss hit! Closing short position")
                self.close_position(sl_price)
    
    def get_equity(self):
        """Get current equity (balance + unrealized P&L)"""
        if self.position == 0:
            return self.balance
        
        # Calculate unrealized P&L
        if self.position > 0:
            unrealized_pnl = (self.current_price - self.entry_price) * self.position * LEVERAGE
        else:
            unrealized_pnl = (self.entry_price - self.current_price) * abs(self.position) * LEVERAGE
        
        return self.balance + unrealized_pnl
    
    def save_trade(self, trade_record):
        """Save trade to JSON log"""
        PAPER_TRADE_LOG.parent.mkdir(exist_ok=True)
        
        # Load existing trades
        existing_trades = []
        if PAPER_TRADE_LOG.exists():
            try:
                with open(PAPER_TRADE_LOG, 'r') as f:
                    existing_trades = json.load(f)
            except:
                existing_trades = []
        
        # Append new trade
        existing_trades.append(trade_record)
        
        # Save back
        with open(PAPER_TRADE_LOG, 'w') as f:
            json.dump(existing_trades, f, indent=2)
    
    def get_stats(self):
        """Get trading statistics"""
        total_trades = len(self.trades)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'balance': self.balance,
                'equity': self.get_equity(),
                'sharpe': 0
            }
        
        win_rate = (self.win_count / total_trades) * 100
        pnl_list = [t['pnl'] for t in self.trades]
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'balance': self.balance,
            'equity': self.get_equity(),
            'avg_pnl': np.mean(pnl_list) if pnl_list else 0,
            'max_pnl': max(pnl_list) if pnl_list else 0,
            'min_pnl': min(pnl_list) if pnl_list else 0
        }


# --- WEBSOCKET PAPER TRADING BOT ---
class PaperTradingBot:
    def __init__(self, loop=None):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("❌ API keys not found in .env file!")
        
        # Capture the primary async event loop running on main thread
        self.loop = loop or asyncio.get_event_loop()
        
        # HTTP client for market data only
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
        
        # Price buffer for indicators (last 100 ticks)
        self.price_buffer = []
        self.bid_buffer = []
        self.ask_buffer = []
        
        # Trade tracking
        self.loss_streak = 0
        self.cooldown_until = 0  # Non-blocking timestamp tracking
        self.is_trading = False
        self.last_trade_time = 0
        self.last_saved_price = 0
        
        # Paper trading state
        self.paper_state = PaperTradingState()
        
        # Threading for concurrent execution
        self.websocket_thread = None
        self.running = False
        
        logger.info(f"✅ Paper trading bot initialized for {SYMBOL}")
        logger.info(f"💰 Initial paper balance: ${PAPER_INITIAL_BALANCE:.2f}")

    # ============================================
    # WEBSOCKET DATA HANDLERS (Thread-Safe)
    # ============================================
    
    def handle_orderbook_data(self, message):
        """Handle real-time order book updates safely from background thread"""
        try:
            data = message.get('data', {})
            if not data:
                return
            
            # Extract bids and asks
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

                # Update paper state with current price (check TP/SL)
                self.paper_state.update_price(self.current_price)

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
    # ASYNC UTILITIES
    # ============================================
    
    async def async_get_paper_balance(self):
        """Get paper trading balance (non-blocking)"""
        return self.paper_state.balance

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

    async def execute_paper_trade(self, side, quantity, tp_price, sl_price):
        """Execute paper trade without real order placement"""
        try:
            # Check if position already exists
            if self.paper_state.position != 0:
                logger.warning("⚠️ Position already open, closing first")
                self.paper_state.close_position(self.current_price)

            # Check if price has moved enough
            min_price_change = 0.000005  # 5 micro-pips
            if abs(self.current_price - self.entry_price) < min_price_change:
                logger.warning(f"Price not moving enough, skipping trade")
                return False
            
            # Open paper position
            entry_price = self.current_price
            self.paper_state.open_position(side, quantity, entry_price, tp_price, sl_price)
            
            self.loss_streak = 0  # Reset streak counter
            
            # Log paper trade
            logger.success(f"✅ [PAPER] Trade executed: {side} {quantity:.4f} @ {entry_price:.8f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Paper trade execution error: {e}")
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
            
            # Get paper balance
            balance = await self.async_get_paper_balance()
            if balance < MIN_BALANCE:
                # Kill the system state switches BEFORE logging
                self.running = False
                
                logger.critical(f"💀 Paper balance threshold breached! Equity: ${balance:.2f} < Minimum: ${MIN_BALANCE:.2f}. Halting execution system.")
                
                if self.ws:
                    self.ws.exit()
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
            
            if rsi > 58 and price_vs_sma > 0.0005:
                side = "Sell"
                tp_price = round(entry_price * (1 - TARGET_PERCENT), 8)
                sl_price = round(entry_price * (1 + STOP_PERCENT), 8)
            elif rsi < 35 and price_vs_sma < -0.0005:
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
            
            # Execute paper trade
            success = await self.execute_paper_trade(side, quantity, tp_price, sl_price)
            
            if success:
                self.last_trade_time = time.time()
                logger.success(f"✅ Paper trade executed!")
            else:
                logger.error(f"❌ Paper trade failed!")
                self.loss_streak += 1
                
            self.is_trading = False
            
        except Exception as e:
            logger.error(f"Trading logic execution error loop crash: {e}")
            self.is_trading = False

    # ============================================
    # PRICE DATA SAVING
    # ============================================
    
    def save_price_data(self):
        """Save price data to CSV for backtesting"""
        import csv
        from pathlib import Path
        
        csv_file = Path(__file__).parent / 'data' / 'paper_price_history.csv'
        csv_file.parent.mkdir(exist_ok=True)
        
        file_exists = csv_file.exists()
        
        # Calculate total volume from top 5 levels
        bid_volume = sum(self.bid_buffer[:5]) if len(self.bid_buffer) >= 5 else 0
        ask_volume = sum(self.ask_buffer[:5]) if len(self.ask_buffer) >= 5 else 0
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'price', 'bid', 'ask', 'spread',
                    'bid_volume', 'ask_volume', 'imbalance'
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
    
    def save_paper_stats(self):
        """Save paper trading statistics to CSV"""
        import csv
        
        stats_file = PAPER_PNL_LOG
        stats_file.parent.mkdir(exist_ok=True)
        
        stats = self.paper_state.get_stats()
        
        # Add timestamp
        stats['timestamp'] = datetime.utcnow().isoformat()
        stats['current_price'] = self.current_price
        stats['position'] = self.paper_state.position
        stats['entry_price'] = self.paper_state.entry_price
        
        file_exists = stats_file.exists()
        
        with open(stats_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)

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
            # Fixed parameter signature rules
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
        logger.info(f"🚀 Starting Paper Trading Bot on {SYMBOL}")
        logger.info(f"   Leverage: {LEVERAGE}x")
        logger.info(f"   Initial Balance: ${PAPER_INITIAL_BALANCE:.2f}")
        
        self.running = True
        
        # Start WebSocket background framework thread
        self.websocket_thread = threading.Thread(target=self.start_websocket, daemon=True)
        self.websocket_thread.start()
        
        # Keep main runtime alive and safely monitor state
        stats_counter = 0
        while self.running:
            await asyncio.sleep(1)
            current_time = int(time.time())
            
            # Monitor balance and save stats every minute
            if current_time % 60 == 0:
                if not self.running:
                    break
                
                balance = await self.async_get_paper_balance()
                equity = self.paper_state.get_equity()
                
                # Get trading stats
                stats = self.paper_state.get_stats()
                
                logger.info(f"💰 [PAPER] Balance: ${balance:.2f} | Equity: ${equity:.2f} | P&L: ${stats['total_pnl']:.2f} | Trades: {stats['total_trades']} | Win Rate: {stats['win_rate']:.1f}%")
                
                # Save stats to CSV
                self.save_paper_stats()
                
                # Check if we should stop
                if balance < MIN_BALANCE:
                    logger.critical(f"💀 Paper balance too low (${balance:.2f}). Stopping system runtime.")
                    self.running = False
                    if self.ws:
                        self.ws.exit()
                    break

        # Print final stats
        final_stats = self.paper_state.get_stats()
        logger.info("=" * 60)
        logger.info("📊 PAPER TRADING FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {final_stats['total_trades']}")
        logger.info(f"Win Rate: {final_stats['win_rate']:.2f}%")
        logger.info(f"Total P&L: ${final_stats['total_pnl']:.2f}")
        logger.info(f"Final Balance: ${final_stats['balance']:.2f}")
        logger.info(f"Final Equity: ${final_stats['equity']:.2f}")
        logger.info("=" * 60)


# ============================================
# ENTRY FILE INTEGRATION
# ============================================

async def main():
    # Pass the active loop reference down to the constructor block
    bot = PaperTradingBot(loop=asyncio.get_running_loop())
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Paper trading bot stopped by user manually.")
    except Exception as e:
        logger.error(f"❌ Fatal error in loop initialization payload: {e}")