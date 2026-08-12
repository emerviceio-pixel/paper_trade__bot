import os
import sys
import asyncio
import json
import time
import threading
import csv
from datetime import datetime, timezone
from pathlib import Path
from decimal import Decimal, ROUND_DOWN
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

# Strategy Parameters
RSI_BUY_THRESHOLD = 35
RSI_SELL_THRESHOLD = 65
SMA_PERIOD = 20
ATR_PERIOD = 14
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0

# --- PAPER TRADING CONFIGURATION ---
PAPER_INITIAL_BALANCE = float(os.getenv('PAPER_INITIAL_BALANCE', 10.0))
PAPER_TRADE_LOG = Path(__file__).parent / 'data' / 'paper_trades.jsonl'
PAPER_PNL_LOG = Path(__file__).parent / 'data' / 'paper_pnl.csv'

class CandleAggregator:
    """Aggregates raw ticks into time-based OHLCV candles for accurate indicators"""
    def __init__(self, interval_sec=60):
        self.interval_sec = interval_sec
        self.candles = deque(maxlen=100)
        self.current_candle = None

    def update(self, price, volume=0, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
            
        candle_time = int(timestamp // self.interval_sec) * self.interval_sec
        
        if self.current_candle is None or self.current_candle['time'] != candle_time:
            if self.current_candle is not None:
                self.candles.append(self.current_candle)
            self.current_candle = {
                'time': candle_time, 'open': price, 'high': price,
                'low': price, 'close': price, 'volume': volume
            }
        else:
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['volume'] += volume
            
    def get_closes(self):
        closes = [c['close'] for c in self.candles]
        if self.current_candle: closes.append(self.current_candle['close'])
        return np.array(closes)
        
    def get_highs(self):
        highs = [c['high'] for c in self.candles]
        if self.current_candle: highs.append(self.current_candle['high'])
        return np.array(highs)

    def get_lows(self):
        lows = [c['low'] for c in self.candles]
        if self.current_candle: lows.append(self.current_candle['low'])
        return np.array(lows)


class PaperTradingState:
    """Maintains paper trading state (position, P&L, history)"""
    def __init__(self, initial_balance=PAPER_INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0  
        self.entry_price = 0.0
        self.current_price = 0.0
        self.trades = []  
        self.open_trade = None  
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.peak_equity = initial_balance
        self.max_drawdown = 0.0
        
    def load_state(self):
        """Load state from JSONL file on startup"""
        if PAPER_TRADE_LOG.exists():
            with open(PAPER_TRADE_LOG, 'r') as f:
                for line in f:
                    try:
                        trade = json.loads(line)
                        self.trades.append(trade)
                        self.balance = trade.get('balance_after', self.balance)
                        self.total_pnl += trade.get('pnl', 0)
                        if trade.get('pnl', 0) > 0:
                            self.win_count += 1
                            self.gross_profit += trade.get('pnl', 0)
                        else:
                            self.loss_count += 1
                            self.gross_loss += abs(trade.get('pnl', 0))
                    except Exception as e:
                        logger.error(f"Error loading trade log: {e}")
            self.peak_equity = max(self.peak_equity, self.balance)
            logger.info(f"📂 Loaded {len(self.trades)} previous trades. Current Balance: ${self.balance:.2f}")

    def open_position(self, side, quantity, entry_price, tp_price, sl_price):
        if self.position != 0:
            logger.warning("⚠️ Position already open, closing first")
            self.close_position(entry_price)
            
        self.position = quantity if side == "Buy" else -quantity
        self.entry_price = entry_price
        self.current_price = entry_price
        
        self.open_trade = {
            'side': side, 'quantity': quantity, 'entry_price': entry_price,
            'tp_price': tp_price, 'sl_price': sl_price,
            'open_time': datetime.now(timezone.utc).isoformat(), 'status': 'open'
        }
        
        logger.info(f"📊 [PAPER] Opened {side} position: {quantity:.4f} @ {entry_price:.8f}")
        logger.info(f"   🎯 TP: {tp_price:.8f} | 🛑 SL: {sl_price:.8f}")
        
    def close_position(self, exit_price):
        if self.position == 0 or not self.open_trade:
            return 0.0
        
        # Calculate P&L
        if self.position > 0:  
            pnl = (exit_price - self.entry_price) * self.position
        else:  
            pnl = (self.entry_price - exit_price) * abs(self.position)
        
        pnl = pnl * LEVERAGE

        # Deduct Bybit taker fees (Entry + Exit)
        FEE_RATE = 0.00055  
        position_value = abs(self.position) * self.entry_price
        fees = position_value * FEE_RATE * 2 * LEVERAGE 
        pnl = pnl - fees
        
        self.balance += pnl
        self.total_pnl += pnl
        
        if pnl > 0:
            self.win_count += 1
            self.gross_profit += pnl
        else:
            self.loss_count += 1
            self.gross_loss += abs(pnl)
            
        # Update Peak Equity and Max Drawdown
        current_equity = self.balance 
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
        # Calculate Return on Margin (ROE)
        margin_used = (self.entry_price * abs(self.position)) / LEVERAGE
        pnl_percent = (pnl / margin_used) * 100 if margin_used > 0 else 0
        
        trade_record = {
            **self.open_trade,
            'exit_price': exit_price,
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'pnl': pnl, 'pnl_percent': pnl_percent,
            'balance_after': self.balance, 'status': 'closed'
        }
        self.trades.append(trade_record)
        
        logger.info(f"📊 [PAPER] Closed position @ {exit_price:.8f}")
        logger.info(f"   P&L: ${pnl:.2f} | Balance: ${self.balance:.2f}")
        
        self.position = 0.0
        self.entry_price = 0.0
        self.open_trade = None
        self.save_trade(trade_record)
        return pnl
    
    def update_price(self, current_price, best_bid, best_ask):
        self.current_price = current_price
        if self.position == 0 or not self.open_trade:
            return
            
        # Realistic Taker Exit Fills
        exit_price = best_bid if self.position > 0 else best_ask
        tp_price = self.open_trade['tp_price']
        sl_price = self.open_trade['sl_price']
        
        if self.position > 0:  # Long
            if exit_price >= tp_price:
                logger.info(f"🎯 [PAPER] Take Profit hit! Closing long position")
                self.close_position(exit_price)
            elif exit_price <= sl_price:
                logger.info(f"🛑 [PAPER] Stop Loss hit! Closing long position")
                worst_case_exit = sl_price - (self.tick_size * 3)
                actual_exit = max(exit_price, worst_case_exit)
                self.close_position(actual_exit)
        else:  # Short
            if exit_price <= tp_price:
                logger.info(f"🎯 [PAPER] Take Profit hit! Closing short position")
                self.close_position(exit_price)
            elif exit_price >= sl_price:
                logger.info(f"🛑 [PAPER] Stop Loss hit! Closing short position")
                worst_case_exit = sl_price + (self.tick_size * 3)
                actual_exit = min(exit_price, worst_case_exit)
                self.close_position(actual_exit)
    
    def get_equity(self):
        if self.position == 0: return self.balance
        if self.position > 0:
            unrealized_pnl = (self.current_price - self.entry_price) * self.position * LEVERAGE
        else:
            unrealized_pnl = (self.entry_price - self.current_price) * abs(self.position) * LEVERAGE
        return self.balance + unrealized_pnl
        
    def save_trade(self, trade_record):
        PAPER_TRADE_LOG.parent.mkdir(exist_ok=True, parents=True)
        with open(PAPER_TRADE_LOG, 'a') as f:
            f.write(json.dumps(trade_record) + '\n')
            
    def get_stats(self):
        total_trades = len(self.trades)
        if total_trades == 0:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'balance': self.balance, 
                    'equity': self.get_equity(), 'max_drawdown': 0, 'profit_factor': 0, 'expectancy': 0}
            
        win_rate = (self.win_count / total_trades) * 100
        pnl_list = [t['pnl'] for t in self.trades]
        profit_factor = self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf')
        expectancy = self.total_pnl / total_trades
        
        return {
            'total_trades': total_trades, 'win_rate': win_rate, 'total_pnl': self.total_pnl,
            'balance': self.balance, 'equity': self.get_equity(),
            'avg_pnl': np.mean(pnl_list) if pnl_list else 0,
            'max_pnl': max(pnl_list) if pnl_list else 0,
            'min_pnl': min(pnl_list) if pnl_list else 0,
            'max_drawdown': self.max_drawdown * 100,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }


class PaperTradingBot:
    def __init__(self, loop=None):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("❌ API keys not found in .env file!")
            
        self.loop = loop or asyncio.get_event_loop()
        self.session = HTTP(testnet=False, api_key=self.api_key, api_secret=self.api_secret)
        
        # Fetch dynamic qty step
        try:
            info = self.session.get_instruments_info(category="linear", symbol=SYMBOL)
            self.qty_step = info['result']['list'][0]['lotSizeFilter']['qtyStep']
            self.tick_size = float(info['result']['list'][0]['priceFilter']['tickSize']) # NEW
            logger.info(f"✅ Fetched Tick Size: {self.tick_size}")
        except Exception as e:
            logger.error(f"Failed to fetch lot step, defaulting to 1: {e}")
            self.tick_size = 0.000001 # Fallback
            
        self.ws = None  
        self.current_price = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.top_bid_vol = 0.0
        self.top_ask_vol = 0.0
        self.last_update_time = 0
        
        # Trade tracking
        self.loss_streak = 0
        self.cooldown_until = 0 
        self.is_trading = False
        self.last_trade_time = 0
        self.last_saved_price = 0
        
        self.candle_agg = CandleAggregator(interval_sec=60) # 1-minute candles
        self._update_scheduled = False # Asyncio throttle flag
        
        self.paper_state = PaperTradingState()
        self.paper_state.load_state()
        
        self.websocket_thread = None
        self.running = False
        
        logger.info(f"✅ Paper trading bot initialized for {SYMBOL}")
        logger.info(f"💰 Initial paper balance: ${self.paper_state.balance:.2f}")

    def handle_orderbook_data(self, message):
        try:
            data = message.get('data', {})
            if not data: return
                
            bids = data.get('b', [])
            asks = data.get('a', [])
            
            if bids:
                self.best_bid = float(bids[0][0])
                self.top_bid_vol = sum(float(b[1]) for b in bids[:5])
            if asks:
                self.best_ask = float(asks[0][0])
                self.top_ask_vol = sum(float(a[1]) for a in asks[:5])
                
            if self.best_bid > 0 and self.best_ask > 0:
                self.current_price = (self.best_bid + self.best_ask) / 2
                self.last_update_time = time.time()
                
                self.candle_agg.update(self.current_price, timestamp=self.last_update_time)
                
                if len(self.candle_agg.candles) % 10 == 0: 
                    self.save_price_data()
                    
                self.paper_state.update_price(self.current_price, self.best_bid, self.best_ask)
                
                # Throttle Async Loop Call
                if not self._update_scheduled:
                    self._update_scheduled = True
                    async def wrapper():
                        await self.on_price_update()
                        self._update_scheduled = False
                    asyncio.run_coroutine_threadsafe(wrapper(), self.loop)
                    
        except Exception as e:
            logger.error(f"Order book handler error: {e}")

    def handle_trade_data(self, message):
        pass # Handled via orderbook for this strategy

    async def execute_paper_trade(self, side, quantity, tp_price, sl_price):
        try:
            if self.paper_state.position != 0:
                logger.warning("⚠️ Position already open, closing first")
                exit_price = self.best_bid if self.paper_state.position > 0 else self.best_ask
                self.paper_state.close_position(exit_price)
                
            # Realistic entry prices (Taker crossing the spread)
            entry_price = self.best_ask if side == "Buy" else self.best_bid
            
            self.paper_state.open_position(side, quantity, entry_price, tp_price, sl_price)
            self.loss_streak = 0 
            
            logger.success(f"✅ [PAPER] Trade executed: {side} {quantity:.4f} @ {entry_price:.8f}")
            return True
            
        except Exception as e:
            logger.error(f"Paper trade execution error: {e}")
            return False

    def calculate_rsi(self, period=14):
        closes = self.candle_agg.get_closes()
        if len(closes) < period + 1: return 50.0
            
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
        
    def calculate_atr(self, period=14):
        closes = self.candle_agg.get_closes()
        highs = self.candle_agg.get_highs()
        lows = self.candle_agg.get_lows()
        
        if len(closes) < period + 1: return 0.0
            
        prev_closes = closes[:-1]
        curr_closes = closes[1:]
        curr_highs = highs[1:]
        curr_lows = lows[1:]
        
        tr1 = curr_highs - curr_lows
        tr2 = np.abs(curr_highs - prev_closes)
        tr3 = np.abs(curr_lows - prev_closes)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.mean(tr[-period:])

    async def on_price_update(self):
        if not self.running or self.is_trading: return
            
        now = time.time()
        if now - self.last_trade_time < 1.0: return
        if now < self.cooldown_until: return
        if len(self.candle_agg.candles) < 20: return
            
        try:
            self.is_trading = True
            
            if self.current_price <= 0 or self.best_bid <= 0 or self.best_ask <= 0:
                self.is_trading = False
                return
                
            spread = (self.best_ask - self.best_bid) / self.best_bid
            if spread > MAX_SPREAD:
                self.is_trading = False
                return
                
            balance = self.paper_state.balance
            if balance < MIN_BALANCE:
                self.running = False
                logger.critical(f"💀 Paper balance threshold breached! Halting.")
                if self.ws: self.ws.exit()
                return
                
            if self.loss_streak >= LOSS_STREAK_LIMIT:
                logger.warning(f"⛔ {self.loss_streak} losses. Cooling down for 10 minutes.")
                self.cooldown_until = now + 600
                self.loss_streak = 0
                self.is_trading = False
                return
                
            # --- INDICATORS ---
            rsi = self.calculate_rsi(14)
            atr = self.calculate_atr(14)
            closes = self.candle_agg.get_closes()
            sma = np.mean(closes[-20:]) if len(closes) >= 20 else self.current_price

            fee_drag_percent = 0.00055 * 2  # 0.11% round trip
            volatility_percent = (atr / self.current_price) if self.current_price > 0 else 0
            if volatility_percent < fee_drag_percent:
                # Market is too quiet; fees will eat all profits
                self.is_trading = False
                return
            
            # Orderbook Imbalance (Top 5 levels)
            total_vol = self.top_bid_vol + self.top_ask_vol
            imbalance = self.top_bid_vol / total_vol if total_vol > 0 else 0.5
            
            side = None
            
            # Strategy Logic
            if rsi < RSI_BUY_THRESHOLD and self.current_price > sma and imbalance > 0.55:
                side = "Buy"
                entry_price = self.best_ask
                # Enforce minimum distance (e.g., at least 5 ticks)
                min_dist = self.tick_size * 5
                tp_dist = max(atr * TP_ATR_MULT, min_dist)
                sl_dist = max(atr * SL_ATR_MULT, min_dist)
                # Calculate and Round to nearest valid tick
                tp_price = round(entry_price + tp_dist, 8)
                tp_price = round(tp_price / self.tick_size) * self.tick_size
                sl_price = round(entry_price - sl_dist, 8)
                sl_price = round(sl_price / self.tick_size) * self.tick_size
            elif rsi > RSI_SELL_THRESHOLD and self.current_price < sma and imbalance < 0.45:
                side = "Sell"
                entry_price = self.best_bid
                # Enforce minimum distance (e.g., at least 5 ticks)
                min_dist = self.tick_size * 5
                tp_dist = max(atr * TP_ATR_MULT, min_dist)
                sl_dist = max(atr * SL_ATR_MULT, min_dist)
                # Calculate and Round to nearest valid tick
                tp_price = round(entry_price - tp_dist, 8)
                tp_price = round(tp_price / self.tick_size) * self.tick_size
                sl_price = round(entry_price + sl_dist, 8)
                sl_price = round(sl_price / self.tick_size) * self.tick_size
            else:
                self.is_trading = False
                return
                
            if not side or atr <= 0:
                self.is_trading = False
                return
                
            logger.info(f"🎯 {side} signal! RSI: {rsi:.1f} | ATR: {atr:.8f} | Imbalance: {imbalance:.2f}")
            
            # Dynamic Lot Sizing
            notional_target = MAX_TRADE_SIZE * LEVERAGE
            quantity = notional_target / entry_price
            
            step_dec = Decimal(str(self.qty_step))
            qty_dec = Decimal(str(quantity)).quantize(step_dec, rounding=ROUND_DOWN)
            quantity = float(qty_dec)
            
            if quantity == 0:
                logger.warning("⚠️ Calculated trade quantity returned 0 lot sizing.")
                self.is_trading = False
                return
                
            success = await self.execute_paper_trade(side, quantity, tp_price, sl_price)
            
            if success:
                self.last_trade_time = time.time()
            else:
                self.loss_streak += 1
                
            self.is_trading = False
            
        except Exception as e:
            logger.error(f"Trading logic error: {e}")
            self.is_trading = False

    def save_price_data(self):
        csv_file = Path(__file__).parent / 'data' / 'paper_price_history.csv'
        csv_file.parent.mkdir(exist_ok=True, parents=True)
        
        file_exists = csv_file.exists()
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'price', 'bid', 'ask', 'spread',
                    'bid_volume', 'ask_volume', 'imbalance'
                ])
            
            spread = (self.best_ask - self.best_bid) / self.best_bid if self.best_bid > 0 else 0
            total_vol = self.top_bid_vol + self.top_ask_vol
            imbalance = self.top_bid_vol / total_vol if total_vol > 0 else 0.5

            if abs(self.current_price - self.last_saved_price) < 0.000001:
                return
            self.last_saved_price = self.current_price
            
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                self.current_price, self.best_bid, self.best_ask,
                spread, self.top_bid_vol, self.top_ask_vol, imbalance
            ])

    def save_paper_stats(self):
        stats_file = PAPER_PNL_LOG
        stats_file.parent.mkdir(exist_ok=True, parents=True)
        stats = self.paper_state.get_stats()
        
        row = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'current_price': self.current_price,
            'position': self.paper_state.position,
            'entry_price': self.paper_state.entry_price,
            'balance': stats['balance'], 'equity': stats['equity'],
            'total_pnl': stats['total_pnl'], 'win_rate': stats['win_rate'],
            'max_drawdown': stats['max_drawdown'], 'profit_factor': stats['profit_factor'],
            'expectancy': stats['expectancy']
        }
        
        file_exists = stats_file.exists()
        with open(stats_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(row)

    def start_websocket(self):
        try:
            logger.info("🔌 Connecting to Bybit WebSocket...")
            self.ws = WebSocket(
                testnet=False, channel_type='linear',
                api_key=self.api_key, api_secret=self.api_secret,
            )
            self.ws.orderbook_stream(50, symbol=SYMBOL, callback=self.handle_orderbook_data)
            self.ws.trade_stream(symbol=SYMBOL, callback=self.handle_trade_data)
            logger.info("✅ WebSocket connected and streaming!")
        except Exception as e:
            logger.error(f"WebSocket error initialization failed: {e}")
            self.running = False

    async def run(self):
        logger.info(f"🚀 Starting Paper Trading Bot on {SYMBOL}")
        logger.info(f"   Leverage: {LEVERAGE}x | Initial Balance: ${PAPER_INITIAL_BALANCE:.2f}")
        
        self.running = True
        self.websocket_thread = threading.Thread(target=self.start_websocket, daemon=True)
        self.websocket_thread.start()
        
        while self.running:
            await asyncio.sleep(1)
            current_time = int(time.time())
            
            if current_time % 60 == 0:
                if not self.running: break
                
                balance = self.paper_state.balance
                equity = self.paper_state.get_equity()
                stats = self.paper_state.get_stats()
                
                logger.info(f"💰 [PAPER] Balance: ${balance:.2f} | Equity: ${equity:.2f} | P&L: ${stats['total_pnl']:.2f} | Trades: {stats['total_trades']} | Win Rate: {stats['win_rate']:.1f}%")
                self.save_paper_stats()
                
                if balance < MIN_BALANCE:
                    logger.critical(f"💀 Paper balance too low (${balance:.2f}). Stopping system runtime.")
                    self.running = False
                    if self.ws: self.ws.exit()
                    break

        final_stats = self.paper_state.get_stats()
        logger.info("=" * 60)
        logger.info("📊 PAPER TRADING FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {final_stats['total_trades']}")
        logger.info(f"Win Rate: {final_stats['win_rate']:.2f}%")
        logger.info(f"Total P&L: ${final_stats['total_pnl']:.2f}")
        logger.info(f"Profit Factor: {final_stats['profit_factor']:.2f}")
        logger.info(f"Max Drawdown: {final_stats['max_drawdown']:.2f}%")
        logger.info(f"Final Balance: ${final_stats['balance']:.2f}")
        logger.info("=" * 60)


async def main():
    bot = PaperTradingBot(loop=asyncio.get_running_loop())
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Paper trading bot stopped by user manually.")
    except Exception as e:
        logger.error(f"❌ Fatal error in loop initialization payload: {e}")