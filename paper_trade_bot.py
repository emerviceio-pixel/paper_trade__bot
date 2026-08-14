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

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from pybit.unified_trading import HTTP, WebSocket
import numpy as np
from loguru import logger

logger.remove()
logger.add(
    "logs/paper_trading.log",
    rotation="1 day", retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO"
)
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG"
)

load_dotenv(Path(__file__).parent / '.env')

# --- CONFIGURATION ---
SYMBOLS = os.getenv('SYMBOLS', '1000PEPEUSDT,BTCUSDT,ETHUSDT').split(',')
SYMBOLS = [s.strip() for s in SYMBOLS]

LEVERAGE = int(os.getenv('LEVERAGE', 5))
MAX_TRADE_SIZE = float(os.getenv('MAX_TRADE_SIZE', 10.0))
MIN_BALANCE = float(os.getenv('MIN_BALANCE', 9.50))
LOSS_STREAK_LIMIT = int(os.getenv('LOSS_STREAK_LIMIT', 3))
MAX_SPREAD = float(os.getenv('MAX_SPREAD', 0.0015))
MAX_CONCURRENT_POSITIONS = int(os.getenv('MAX_CONCURRENT_POSITIONS', 3))

# Strategy
RSI_BUY_THRESHOLD = 35
RSI_SELL_THRESHOLD = 65
SMA_PERIOD = 20
ATR_PERIOD = 14
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0
MIN_TICK_DISTANCE = 5  # Minimum ticks for TP/SL

# Paper Trading
PAPER_INITIAL_BALANCE = float(os.getenv('PAPER_INITIAL_BALANCE', 100.0))
PAPER_TRADE_LOG = Path(__file__).parent / 'data' / 'paper_trades.jsonl'
PAPER_PNL_LOG = Path(__file__).parent / 'data' / 'paper_pnl.csv'
SIGNAL_LOG = Path(__file__).parent / 'data' / 'signals.csv'


# ============================================================
# CANDLE AGGREGATOR (unchanged)
# ============================================================
class CandleAggregator:
    def __init__(self, interval_sec=60):
        self.interval_sec = interval_sec
        self.candles = deque(maxlen=200)
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
        if self.current_candle:
            closes.append(self.current_candle['close'])
        return np.array(closes)

    def get_highs(self):
        highs = [c['high'] for c in self.candles]
        if self.current_candle:
            highs.append(self.current_candle['high'])
        return np.array(highs)

    def get_lows(self):
        lows = [c['low'] for c in self.candles]
        if self.current_candle:
            lows.append(self.current_candle['low'])
        return np.array(lows)


# ============================================================
# PER-SYMBOL TRACKER (NEW)
# ============================================================
class SymbolTracker:
    """Isolates all market data, indicators, and signal state per symbol."""

    def __init__(self, symbol: str, tick_size: float, qty_step: str):
        self.symbol = symbol
        self.tick_size = tick_size
        self.qty_step = qty_step

        # Orderbook state
        self.current_price = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.top_bid_vol = 0.0
        self.top_ask_vol = 0.0
        self.last_update_time = 0

        # Indicators
        self.candle_agg = CandleAggregator(interval_sec=60)

        # Signal state (for logging even when not trading)
        self.last_rsi = 50.0
        self.last_atr = 0.0
        self.last_sma = 0.0
        self.last_imbalance = 0.5
        self.last_signal = None
        self.last_signal_time = 0

    def update_orderbook(self, bids, asks):
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

    @property
    def spread(self):
        if self.best_bid <= 0:
            return float('inf')
        return (self.best_ask - self.best_bid) / self.best_bid

    @property
    def imbalance(self):
        total = self.top_bid_vol + self.top_ask_vol
        return self.top_bid_vol / total if total > 0 else 0.5

    def calculate_rsi(self, period=14):
        closes = self.candle_agg.get_closes()
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def calculate_atr(self, period=14):
        closes = self.candle_agg.get_closes()
        highs = self.candle_agg.get_highs()
        lows = self.candle_agg.get_lows()
        if len(closes) < period + 1:
            return 0.0
        prev_closes = closes[:-1]
        curr_highs = highs[1:]
        curr_lows = lows[1:]
        tr1 = curr_highs - curr_lows
        tr2 = np.abs(curr_highs - prev_closes)
        tr3 = np.abs(curr_lows - prev_closes)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.mean(tr[-period:])

    def update_indicators(self):
        """Recalculate all indicators. Call periodically, not every tick."""
        self.last_rsi = self.calculate_rsi(RSI_PERIOD if 'RSI_PERIOD' in dir() else 14)
        self.last_atr = self.calculate_atr(ATR_PERIOD)
        closes = self.candle_agg.get_closes()
        self.last_sma = np.mean(closes[-SMA_PERIOD:]) if len(closes) >= SMA_PERIOD else self.current_price
        self.last_imbalance = self.imbalance

    def get_signal(self):
        """Evaluate trading signal. Returns 'Buy', 'Sell', or None."""
        if len(self.candle_agg.candles) < SMA_PERIOD:
            return None

        # Fee drag filter
        if self.current_price > 0 and self.last_atr > 0:
            vol_pct = self.last_atr / self.current_price
            if vol_pct < (0.00055 * 2):  # Below round-trip fee cost
                return None

        if self.last_rsi < RSI_BUY_THRESHOLD and self.current_price > self.last_sma and self.last_imbalance > 0.55:
            return "Buy"
        elif self.last_rsi > RSI_SELL_THRESHOLD and self.current_price < self.last_sma and self.last_imbalance < 0.45:
            return "Sell"
        return None

    def calculate_tp_sl(self, side, entry_price):
        """Calculate tick-aligned TP/SL with minimum distance enforcement."""
        min_dist = self.tick_size * MIN_TICK_DISTANCE
        spread_dist = self.best_ask - self.best_bid  # Current spread
        # SL must be at least spread + buffer away
        sl_dist = max(self.last_atr * SL_ATR_MULT, min_dist, spread_dist * 1.5)
        tp_dist = max(self.last_atr * TP_ATR_MULT, min_dist)

        if side == "Buy":
            tp_raw = entry_price + tp_dist
            sl_raw = entry_price - sl_dist
        else:
            tp_raw = entry_price - tp_dist
            sl_raw = entry_price + sl_dist

        # Round to valid tick size
        tp_price = round(round(tp_raw, 10) / self.tick_size) * self.tick_size
        sl_price = round(round(sl_raw, 10) / self.tick_size) * self.tick_size
        return tp_price, sl_price


# ============================================================
# MULTI-POSITION PAPER STATE (NEW)
# ============================================================
class MultiPositionState:
    """Manages multiple concurrent positions with a shared balance."""

    def __init__(self, initial_balance=PAPER_INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # symbol -> position dict
        self.trades = []
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.peak_equity = initial_balance
        self.max_drawdown = 0.0

    def load_state(self):
        if PAPER_TRADE_LOG.exists():
            with open(PAPER_TRADE_LOG, 'r') as f:
                for line in f:
                    try:
                        trade = json.loads(line)
                        self.trades.append(trade)
                        self.balance = trade.get('balance_after', self.balance)
                        pnl = trade.get('pnl', 0)
                        self.total_pnl += pnl
                        if pnl > 0:
                            self.win_count += 1
                            self.gross_profit += pnl
                        else:
                            self.loss_count += 1
                            self.gross_loss += abs(pnl)
                    except Exception:
                        pass
            self.peak_equity = max(self.peak_equity, self.balance)
            logger.info(f"📂 Loaded {len(self.trades)} trades. Balance: ${self.balance:.2f}")

    def open_position(self, symbol, side, quantity, entry_price, tp_price, sl_price):
        if symbol in self.positions:
            logger.warning(f"⚠️ [{symbol}] Position already open, closing first")
            self.close_position(symbol, entry_price)

        self.positions[symbol] = {
            'side': side, 'quantity': quantity, 'entry_price': entry_price,
            'tp_price': tp_price, 'sl_price': sl_price,
            'open_time': datetime.now(timezone.utc).isoformat(), 'status': 'open'
        }
        logger.info(f"📊 [{symbol}] Opened {side}: {quantity:.4f} @ {entry_price:.8f} | TP: {tp_price:.8f} | SL: {sl_price:.8f}")

    def close_position(self, symbol, exit_price):
        if symbol not in self.positions:
            return 0.0

        pos = self.positions[symbol]
        quantity = pos['quantity']
        entry_price = pos['entry_price']

        if pos['side'] == "Buy":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity


        # If losing trade, apply a symbol-specific cooldown
        if pnl < 0:
            self.symbol_cooldowns[symbol] = time.time() + 60  # 60s cooldown

        # Fees
        position_value = quantity * entry_price
        fees = position_value * 0.00055 * 2 * LEVERAGE
        pnl -= fees

        self.balance += pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.win_count += 1
            self.gross_profit += pnl
        else:
            self.loss_count += 1
            self.gross_loss += abs(pnl)

        # Drawdown
        equity = self.get_equity()
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        margin_used = (entry_price * quantity) / LEVERAGE
        pnl_percent = (pnl / margin_used) * 100 if margin_used > 0 else 0

        trade_record = {
            **pos, 'symbol': symbol,
            'exit_price': exit_price,
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'pnl': pnl, 'pnl_percent': pnl_percent,
            'balance_after': self.balance, 'status': 'closed'
        }
        self.trades.append(trade_record)
        del self.positions[symbol]

        logger.info(f"📊 [{symbol}] Closed @ {exit_price:.8f} | P&L: ${pnl:.4f} | Balance: ${self.balance:.2f}")
        self._save_trade(trade_record)
        return pnl

    def check_tp_sl(self, symbol, best_bid, best_ask):
        """Check if TP/SL is hit for a given symbol."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        exit_price = best_bid if pos['side'] == "Buy" else best_ask

        if pos['side'] == "Buy":
            if exit_price >= pos['tp_price']:
                logger.info(f"🎯 [{symbol}] TP hit!")
                self.close_position(symbol, exit_price)
            elif exit_price <= pos['sl_price']:
                logger.info(f"🛑 [{symbol}] SL hit!")
                self.close_position(symbol, exit_price)
        else:
            if exit_price <= pos['tp_price']:
                logger.info(f"🎯 [{symbol}] TP hit!")
                self.close_position(symbol, exit_price)
            elif exit_price >= pos['sl_price']:
                logger.info(f"🛑 [{symbol}] SL hit!")
                self.close_position(symbol, exit_price)

    def get_equity(self):
        equity = self.balance
        for symbol, pos in self.positions.items():
            # Use mid-price approximation for unrealized P&L
            if pos['side'] == "Buy":
                unrealized = (pos.get('current_price', pos['entry_price']) - pos['entry_price']) * pos['quantity'] * LEVERAGE
            else:
                unrealized = (pos['entry_price'] - pos.get('current_price', pos['entry_price'])) * pos['quantity'] * LEVERAGE
            equity += unrealized
        return equity

    def update_current_price(self, symbol, price):
        if symbol in self.positions:
            self.positions[symbol]['current_price'] = price

    def _save_trade(self, record):
        PAPER_TRADE_LOG.parent.mkdir(exist_ok=True, parents=True)
        with open(PAPER_TRADE_LOG, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def get_stats(self):
        total = len(self.trades)
        if total == 0:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                    'balance': self.balance, 'equity': self.get_equity(),
                    'max_drawdown': 0, 'profit_factor': 0, 'expectancy': 0,
                    'open_positions': len(self.positions)}
        pf = self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf')
        return {
            'total_trades': total,
            'win_rate': (self.win_count / total) * 100,
            'total_pnl': self.total_pnl,
            'balance': self.balance,
            'equity': self.get_equity(),
            'max_drawdown': self.max_drawdown * 100,
            'profit_factor': pf,
            'expectancy': self.total_pnl / total,
            'open_positions': len(self.positions)
        }


# ============================================================
# MULTI-SYMBOL BOT
# ============================================================
class MultiSymbolBot:
    def __init__(self, loop=None):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY')
        if not self.api_key or not self.api_secret:
            raise ValueError("❌ API keys not found in .env!")

        self.loop = loop or asyncio.get_event_loop()
        self.session = HTTP(testnet=False, api_key=self.api_key, api_secret=self.api_secret)

        # Initialize per-symbol trackers
        self.trackers = {}
        for symbol in SYMBOLS:
            try:
                info = self.session.get_instruments_info(category="linear", symbol=symbol)
                inst = info['result']['list'][0]
                tick_size = float(inst['priceFilter']['tickSize'])
                qty_step = inst['lotSizeFilter']['qtyStep']
                self.trackers[symbol] = SymbolTracker(symbol, tick_size, qty_step)
                logger.info(f"✅ [{symbol}] tick={tick_size} | qty_step={qty_step}")
            except Exception as e:
                logger.error(f"❌ Failed to init {symbol}: {e}")

        self.state = MultiPositionState()
        self.state.load_state()

        self.symbol_cooldowns = {}  # symbol -> cooldown timestamp
        self.ws = None
        self.running = False
        self.is_trading = False
        self.cooldown_until = 0
        self.loss_streak = 0
        self.last_trade_time = 0
        self._update_scheduled = False
        self._indicator_counter = 0

    # --- WebSocket Handler ---
    def handle_orderbook(self, message):
        try:
            symbol = message.get('topic', '').replace('orderbook.50.', '')
            if symbol not in self.trackers:
                return

            tracker = self.trackers[symbol]
            data = message.get('data', {})
            tracker.update_orderbook(data.get('b', []), data.get('a', []))

            # Check TP/SL for open positions
            self.state.update_current_price(symbol, tracker.current_price)
            self.state.check_tp_sl(symbol, tracker.best_bid, tracker.best_ask)

            # Throttle async logic
            if not self._update_scheduled:
                self._update_scheduled = True
                async def wrapper():
                    await self.on_tick()
                    self._update_scheduled = False
                asyncio.run_coroutine_threadsafe(wrapper(), self.loop)

        except Exception as e:
            logger.error(f"Orderbook handler error: {e}")

    # --- Main Trading Logic ---
    async def on_tick(self):
        if not self.running or self.is_trading:
            return

        now = time.time()
        if now - self.last_trade_time < 1.0:
            return
        if now < self.cooldown_until:
            return

        try:
            self.is_trading = True

            # Update indicators every 10 ticks to save CPU
            self._indicator_counter += 1
            if self._indicator_counter % 10 == 0:
                for tracker in self.trackers.values():
                    tracker.update_indicators()

            # Check balance
            if self.state.balance < MIN_BALANCE:
                logger.critical("💀 Balance too low. Halting.")
                self.running = False
                return

            # Cooldown check
            if self.loss_streak >= LOSS_STREAK_LIMIT:
                logger.warning(f"⛔ {self.loss_streak} losses. Cooling down 10 min.")
                self.cooldown_until = now + 600
                self.loss_streak = 0
                self.is_trading = False
                return

            # Scan all symbols for signals
            for symbol, tracker in self.trackers.items():
                # Skip if symbol is in cooldown
                if symbol in self.symbol_cooldowns and now < self.symbol_cooldowns[symbol]:
                    continue

                # Skip if already have position in this symbol
                if symbol in self.state.positions:
                    continue

                # Skip if max concurrent positions reached
                if len(self.state.positions) >= MAX_CONCURRENT_POSITIONS:
                    break

                # Skip if spread too wide
                if tracker.spread > MAX_SPREAD:
                    continue

                signal = tracker.get_signal()
                if signal is None:
                    continue

                # Execute trade
                entry_price = tracker.best_ask if signal == "Buy" else tracker.best_bid
                tp_price, sl_price = tracker.calculate_tp_sl(signal, entry_price)

                # Validate SL against market spread
                if signal == "Sell":
                    # For shorts, SL must be ABOVE best_ask (not just above entry)
                    if sl_price <= tracker.best_ask:
                        continue  # Skip, SL is inside spread
                elif signal == "Buy":
                    # For longs, SL must be BELOW best_bid
                    if sl_price >= tracker.best_bid:
                        continue

                # Log signal
                self._log_signal(symbol, tracker, signal)

                notional = MAX_TRADE_SIZE * LEVERAGE
                quantity = notional / entry_price
                qty_dec = Decimal(str(quantity)).quantize(Decimal(str(tracker.qty_step)), rounding=ROUND_DOWN)
                quantity = float(qty_dec)

                if quantity <= 0:
                    continue

                self.state.open_position(symbol, signal, quantity, entry_price, tp_price, sl_price)
                self.last_trade_time = time.time()
                self.loss_streak = 0
                logger.success(f"✅ [{symbol}] {signal} executed!")
                break  # One trade per tick

            self.is_trading = False

        except Exception as e:
            logger.error(f"Trading logic error: {e}")
            self.is_trading = False

    def _log_signal(self, symbol, tracker, signal):
        SIGNAL_LOG.parent.mkdir(exist_ok=True, parents=True)
        file_exists = SIGNAL_LOG.exists()
        with open(SIGNAL_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'symbol', 'signal', 'price', 'rsi', 'atr', 'sma', 'imbalance', 'spread'])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(), symbol, signal,
                tracker.current_price, f"{tracker.last_rsi:.2f}",
                f"{tracker.last_atr:.8f}", f"{tracker.last_sma:.8f}",
                f"{tracker.last_imbalance:.4f}", f"{tracker.spread:.6f}"
            ])

    def save_stats(self):
        stats = self.state.get_stats()
        PAPER_PNL_LOG.parent.mkdir(exist_ok=True, parents=True)
        file_exists = PAPER_PNL_LOG.exists()

        row = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'balance': stats['balance'], 'equity': stats['equity'],
            'total_pnl': stats['total_pnl'], 'win_rate': stats['win_rate'],
            'total_trades': stats['total_trades'],
            'open_positions': stats['open_positions'],
            'max_drawdown': stats['max_drawdown'],
            'profit_factor': stats['profit_factor'],
        }
        with open(PAPER_PNL_LOG, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # --- WebSocket Setup ---
    def start_websocket(self):
        try:
            self.ws = WebSocket(testnet=False, channel_type='linear',
                                api_key=self.api_key, api_secret=self.api_secret)
            for symbol in self.trackers:
                self.ws.orderbook_stream(50, symbol=symbol, callback=self.handle_orderbook)
                logger.info(f"🔌 Subscribed to {symbol} orderbook")
            logger.info("✅ All WebSocket streams active!")
        except Exception as e:
            logger.error(f"WebSocket init failed: {e}")
            self.running = False

    # --- Main Loop ---
    async def run(self):
        logger.info(f"🚀 Multi-Symbol Bot: {', '.join(SYMBOLS)}")
        logger.info(f"   Leverage: {LEVERAGE}x | Max Positions: {MAX_CONCURRENT_POSITIONS}")
        logger.info(f"   Balance: ${self.state.balance:.2f}")

        self.running = True
        ws_thread = threading.Thread(target=self.start_websocket, daemon=True)
        ws_thread.start()

        while self.running:
            await asyncio.sleep(1)
            if int(time.time()) % 60 == 0:
                stats = self.state.get_stats()
                open_syms = list(self.state.positions.keys())
                logger.info(
                    f"💰 Bal: ${stats['balance']:.2f} | Equity: ${stats['equity']:.2f} | "
                    f"P&L: ${stats['total_pnl']:.2f} | Trades: {stats['total_trades']} | "
                    f"Open: {open_syms}"
                )
                self.save_stats()

                if self.state.balance < MIN_BALANCE:
                    self.running = False
                    break

        # Final report
        stats = self.state.get_stats()
        logger.info("=" * 60)
        logger.info("📊 FINAL STATISTICS")
        logger.info(f"   Trades: {stats['total_trades']} | Win Rate: {stats['win_rate']:.1f}%")
        logger.info(f"   P&L: ${stats['total_pnl']:.2f} | Balance: ${stats['balance']:.2f}")
        logger.info(f"   Max DD: {stats['max_drawdown']:.2f}% | PF: {stats['profit_factor']:.2f}")
        logger.info("=" * 60)


async def main():
    bot = MultiSymbolBot(loop=asyncio.get_running_loop())
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user.")
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")