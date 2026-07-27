import os
import sys
import asyncio
import datetime
import json
from pathlib import Path
from decimal import Decimal
import csv
from datetime import datetime

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
SYMBOL = os.getenv('SYMBOL', 'PEPEUSDT')
LEVERAGE = int(os.getenv('LEVERAGE', 5))
MAX_TRADE_SIZE = float(os.getenv('MAX_TRADE_SIZE', 10.0))
MIN_BALANCE = float(os.getenv('MIN_BALANCE', 9.50))
LOSS_STREAK_LIMIT = int(os.getenv('LOSS_STREAK_LIMIT', 2))
MAX_SPREAD = float(os.getenv('MAX_SPREAD', 0.0015))
TRADING_START_HOUR = int(os.getenv('TRADING_START_HOUR', 8))
TRADING_END_HOUR = int(os.getenv('TRADING_END_HOUR', 17))

TARGET_PERCENT = 0.0015  # 0.15%
STOP_PERCENT = 0.0012    # 0.12%

# --- BYBIT CLIENT ---
class BybitBot:
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("❌ API keys not found in .env file!")
        
        # Initialize HTTP client
        self.session = HTTP(
            testnet=False,  # Set to True for testnet
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        
        # Price buffer for indicators
        self.price_buffer = []
        self.loss_streak = 0
        
        logger.info(f"✅ Bybit bot initialized for {SYMBOL}")

        # CSV file setup
    self.csv_file = Path(__file__).parent.parent / 'data' / 'price_history.csv'
    self._init_csv()
    
    def _init_csv(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'price', 'bid_volume', 'ask_volume', 
                            'spread', 'trend', 'side', 'quantity', 'result'])

    def is_trading_hours(self):
        """Check if current time is within trading hours"""
        now = datetime.datetime.utcnow().hour
        return TRADING_START_HOUR <= now < TRADING_END_HOUR

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

    def get_order_book(self):
        """Fetch order book depth"""
        try:
            depth = self.session.get_orderbook(
                category="linear",
                symbol=SYMBOL,
                limit=10
            )
            bids = [[float(b[0]), float(b[1])] for b in depth['result']['b']]
            asks = [[float(a[0]), float(a[1])] for a in depth['result']['a']]
            return bids, asks
        except Exception as e:
            logger.error(f"Order book error: {e}")
            return [], []

    def calculate_spread(self, bids, asks):
        """Calculate spread percentage"""
        if not bids or not asks:
            return 999.0
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        return (best_ask - best_bid) / best_bid

    def calculate_imbalance(self, bids, asks):
        """Calculate order book imbalance (top 5 levels)"""
        bid_volume = sum(b[1] for b in bids[:5])
        ask_volume = sum(a[1] for a in asks[:5])
        total = bid_volume + ask_volume
        if total == 0:
            return 0.5
        return bid_volume / total

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

    def get_trend(self):
        """Detect market trend"""
        if len(self.price_buffer) < 50:
            return "STABLE"
        
        sma = np.mean(self.price_buffer[-50:])
        current_price = self.price_buffer[-1]
        price_vs_sma = current_price - sma
        
        rsi = self.calculate_rsi(14)
        volatility = np.std(self.price_buffer[-20:])
        
        # If volatility is too low, it's stable
        if volatility < (sma * 0.001):
            return "STABLE"
        
        # Voting system
        bullish_votes = 0
        bearish_votes = 0
        
        if price_vs_sma > (sma * 0.002):
            bullish_votes += 1
        elif price_vs_sma < -(sma * 0.002):
            bearish_votes += 1
        
        if rsi > 58:
            bullish_votes += 1
        elif rsi < 42:
            bearish_votes += 1
        
        if bullish_votes > bearish_votes:
            return "BULLISH"
        elif bearish_votes > bullish_votes:
            return "BEARISH"
        else:
            return "STABLE"

    def place_market_order(self, side, quantity):
        """Place a market order"""
        try:
            order = self.session.place_order(
                category="linear",
                symbol=SYMBOL,
                side=side,
                orderType="Market",
                qty=str(quantity),
                timeInForce="GTC"
            )
            return order
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return None

    def place_limit_order(self, side, quantity, price):
        """Place a limit order (for TP)"""
        try:
            order = self.session.place_order(
                category="linear",
                symbol=SYMBOL,
                side=side,
                orderType="Limit",
                qty=str(quantity),
                price=str(price),
                timeInForce="GTC"
            )
            return order
        except Exception as e:
            logger.error(f"Limit order error: {e}")
            return None

    def place_stop_loss(self, side, quantity, stop_price):
        """Place a stop loss (market)"""
        try:
            order = self.session.place_order(
                category="linear",
                symbol=SYMBOL,
                side=side,
                orderType="Market",
                qty=str(quantity),
                timeInForce="GTC",
                triggerPrice=str(stop_price),
                triggerDirection=1 if side == "Sell" else 2  # 1: fall, 2: rise
            )
            return order
        except Exception as e:
            logger.error(f"Stop loss error: {e}")
            return None

    def get_position(self):
        """Get current position"""
        try:
            pos = self.session.get_positions(
                category="linear",
                symbol=SYMBOL
            )
            for p in pos['result']['list']:
                if float(p['size']) != 0:
                    return p
            return None
        except Exception as e:
            logger.error(f"Position error: {e}")
            return None

    def cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            self.session.cancel_all_orders(
                category="linear",
                symbol=SYMBOL
            )
            return True
        except Exception as e:
            logger.error(f"Cancel orders error: {e}")
            return False

    def close_position(self):
        """Close position with market order"""
        position = self.get_position()
        if not position:
            return True
        
        side = "Sell" if float(position['side']) == 1 else "Buy"
        quantity = abs(float(position['size']))
        
        try:
            self.session.place_order(
                category="linear",
                symbol=SYMBOL,
                side=side,
                orderType="Market",
                qty=str(quantity),
                timeInForce="GTC"
            )
            logger.info(f"✅ Position closed: {quantity} {SYMBOL}")
            return True
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return False

    async def execute_trade(self):
        """Main trading logic"""
        # --- SAFETY CHECKS ---
        
        # 1. Trading hours check
        if not self.is_trading_hours():
            logger.info("⏰ Outside trading hours. Sleeping.")
            return False
        
        # 2. Balance check
        balance = self.get_balance()
        if balance < MIN_BALANCE:
            logger.critical(f"💀 Balance {balance} < ${MIN_BALANCE}. Terminating.")
            return False
        
        # 3. Loss streak check
        if self.loss_streak >= LOSS_STREAK_LIMIT:
            logger.warning(f"⛔ {self.loss_streak} losses. Cooling down 10 min.")
            return False
        
        # --- MARKET DATA ---
        bids, asks = self.get_order_book()
        if not bids or not asks:
            logger.warning("No order book data. Skipping.")
            return False
        
        current_price = bids[0][0]
        self.price_buffer.append(current_price)
        if len(self.price_buffer) > 100:
            self.price_buffer.pop(0)
        
        # 4. Spread check
        spread = self.calculate_spread(bids, asks)
        if spread > MAX_SPREAD:
            logger.warning(f"Spread {spread:.4%} > {MAX_SPREAD:.4%}. Skipping.")
            return False
        
        # 5. Imbalance check
        imbalance = self.calculate_imbalance(bids, asks)
        if 0.45 < imbalance < 0.55:
            logger.info("➖ Order book balanced. Skipping.")
            return False
        
        # 6. Trend check
        trend = self.get_trend()
        logger.info(f"📊 Trend: {trend}, Imbalance: {imbalance:.2%}, Spread: {spread:.4%}")
        
        # --- DECISION ---
        if imbalance > 0.55:
            side = "Buy"
            entry_price = current_price
            tp_price = round(entry_price * (1 + TARGET_PERCENT), 8)
            sl_price = round(entry_price * (1 - STOP_PERCENT), 8)
            close_side = "Sell"
        elif imbalance < 0.45:
            side = "Sell"
            entry_price = current_price
            tp_price = round(entry_price * (1 - TARGET_PERCENT), 8)
            sl_price = round(entry_price * (1 + STOP_PERCENT), 8)
            close_side = "Buy"
        else:
            logger.info("No clear signal. Skipping.")
            return False
        
        # --- CALCULATE QUANTITY ---
        quantity = (MAX_TRADE_SIZE * LEVERAGE) / entry_price
        quantity = round(quantity, 0)  # Bybit requires integer for PEPE
        
        if quantity == 0:
            logger.error("Quantity is 0. Skipping.")
            return False
        
        logger.info(f"🎯 {side} {quantity} {SYMBOL} @ {entry_price}")
        logger.info(f"  TP: {tp_price} | SL: {sl_price}")

        # --- WRITE TO CSV ---
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    current_price,
                    sum(b[1] for b in bids[:5]),
                    sum(a[1] for a in asks[:5]),
                    spread,
                    trend,
                    side,
                    quantity,
                    "WIN" if success else "LOSS"
                ])
        except Exception as e:
            logger.warning(f"CSV write error: {e}")
        
        # --- PLACE ORDERS ---
        # 1. Market entry
        entry = self.place_market_order(side, quantity)
        if not entry:
            logger.error("Entry failed.")
            return False
        
        logger.info(f"✅ Entry placed. Order ID: {entry['result']['orderId']}")
        
        # 2. Place take-profit (limit)
        tp = self.place_limit_order(close_side, quantity, tp_price)
        if not tp:
            logger.warning("TP placement failed. Will try to close manually.")
        
        # 3. Place stop-loss (market)
        sl = self.place_stop_loss(close_side, quantity, sl_price)
        if not sl:
            logger.warning("SL placement failed. Will try to close manually.")
        
        # --- WAIT AND MONITOR ---
        await asyncio.sleep(5)
        
        # --- CHECK POSITION ---
        position = self.get_position()
        if position:
            logger.warning("Position still open. Closing manually.")
            self.close_position()
            self.loss_streak += 1
            return False
        else:
            logger.success(f"✅ Trade complete! Profit captured.")
            self.loss_streak = 0
            return True

    async def run(self):
        """Main loop"""
        logger.info(f"🚀 Starting Bybit bot on {SYMBOL}")
        logger.info(f"   Balance: ${self.get_balance():.2f}")
        logger.info(f"   Leverage: {LEVERAGE}x")
        logger.info(f"   Trading hours: {TRADING_START_HOUR}:00 - {TRADING_END_HOUR}:00 UTC")
        
        while True:
            try:
                # Cancel all old orders first
                self.cancel_all_orders()
                
                # Execute one trade
                await self.execute_trade()
                
                # Cooldown
                await asyncio.sleep(2)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                await asyncio.sleep(5)

# --- MAIN ---
async def main():
    bot = BybitBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())