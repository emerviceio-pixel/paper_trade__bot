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

# ── Logging ──────────────────────────────────────────────────
logger.remove()
logger.add(
    "logs/paper_trading.log",
    rotation="1 day", retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
)

load_dotenv(Path(__file__).parent / ".env")

# ── Configuration ────────────────────────────────────────────
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "1000PEPEUSDT,BTCUSDT,ETHUSDT").split(",")]
LEVERAGE           = int(os.getenv("LEVERAGE", 5))
MAX_TRADE_SIZE     = float(os.getenv("MAX_TRADE_SIZE", 10.0))
MIN_BALANCE        = float(os.getenv("MIN_BALANCE", 5.0))
LOSS_STREAK_LIMIT  = int(os.getenv("LOSS_STREAK_LIMIT", 3))
MAX_SPREAD         = float(os.getenv("MAX_SPREAD", 0.0015))
MAX_CONCURRENT     = int(os.getenv("MAX_CONCURRENT_POSITIONS", 3))
PAPER_INITIAL_BAL  = float(os.getenv("PAPER_INITIAL_BALANCE", 100.0))

# Strategy
RSI_BUY_THRESH  = 35
RSI_SELL_THRESH = 65
SMA_PERIOD      = 20
ATR_PERIOD      = 14
SL_ATR_MULT     = 1.5
TP_ATR_MULT     = 2.0
MIN_TICK_DIST   = 5          # minimum ticks for TP / SL
MAX_SLIP_TICKS  = 3          # max slippage ticks on SL fill
SYMBOL_COOLDOWN = 60         # seconds before re-entering after a loss

# Paths
DATA_DIR       = Path(__file__).parent / "data"
TRADE_LOG      = DATA_DIR / "paper_trades.jsonl"
PNL_LOG        = DATA_DIR / "paper_pnl.csv"
SIGNAL_LOG     = DATA_DIR / "signals.csv"


# ════════════════════════════════════════════════════════════
#  CANDLE AGGREGATOR
# ════════════════════════════════════════════════════════════
class CandleAggregator:
    """Builds OHLCV candles from raw ticks."""

    def __init__(self, interval_sec: int = 60):
        self.interval_sec = interval_sec
        self.candles: deque = deque(maxlen=200)
        self.current_candle: dict | None = None

    def update(self, price: float, volume: float = 0.0, ts: float | None = None):
        if ts is None:
            ts = time.time()
        candle_time = int(ts // self.interval_sec) * self.interval_sec

        if self.current_candle is None or self.current_candle["time"] != candle_time:
            if self.current_candle is not None:
                self.candles.append(self.current_candle)
            self.current_candle = {
                "time": candle_time, "open": price, "high": price,
                "low": price, "close": price, "volume": volume,
            }
        else:
            c = self.current_candle
            c["high"] = max(c["high"], price)
            c["low"]  = min(c["low"], price)
            c["close"] = price
            c["volume"] += volume

    def _series(self, key: str) -> np.ndarray:
        vals = [c[key] for c in self.candles]
        if self.current_candle:
            vals.append(self.current_candle[key])
        return np.array(vals, dtype=float)

    def get_closes(self): return self._series("close")
    def get_highs(self):  return self._series("high")
    def get_lows(self):   return self._series("low")


# ════════════════════════════════════════════════════════════
#  PER-SYMBOL TRACKER
# ════════════════════════════════════════════════════════════
class SymbolTracker:
    """Isolates market data, indicators and signal logic per symbol."""

    def __init__(self, symbol: str, tick_size: float, qty_step: str):
        self.symbol    = symbol
        self.tick_size = tick_size
        self.qty_step  = qty_step

        self.current_price = 0.0
        self.best_bid      = 0.0
        self.best_ask      = 0.0
        self.top_bid_vol   = 0.0
        self.top_ask_vol   = 0.0
        self.last_update   = 0.0

        self.candle_agg = CandleAggregator(interval_sec=60)

        # Cached indicator values (updated periodically)
        self.last_rsi       = 50.0
        self.last_atr       = 0.0
        self.last_sma       = 0.0
        self.last_imbalance = 0.5

    # ── orderbook ────────────────────────────────────────────
    def update_orderbook(self, bids: list, asks: list):
        if bids:
            self.best_bid    = float(bids[0][0])
            self.top_bid_vol = sum(float(b[1]) for b in bids[:5])
        if asks:
            self.best_ask    = float(asks[0][0])
            self.top_ask_vol = sum(float(a[1]) for a in asks[:5])

        if self.best_bid > 0 and self.best_ask > 0:
            self.current_price = (self.best_bid + self.best_ask) / 2.0
            self.last_update   = time.time()
            self.candle_agg.update(self.current_price, ts=self.last_update)

    @property
    def spread(self) -> float:
        return (self.best_ask - self.best_bid) / self.best_bid if self.best_bid > 0 else float("inf")

    @property
    def imbalance(self) -> float:
        total = self.top_bid_vol + self.top_ask_vol
        return self.top_bid_vol / total if total > 0 else 0.5

    # ── indicators ───────────────────────────────────────────
    def refresh_indicators(self):
        self.last_rsi       = self._rsi(ATR_PERIOD)
        self.last_atr       = self._atr(ATR_PERIOD)
        closes              = self.candle_agg.get_closes()
        self.last_sma       = float(np.mean(closes[-SMA_PERIOD:])) if len(closes) >= SMA_PERIOD else self.current_price
        self.last_imbalance = self.imbalance

    def _rsi(self, period: int = 14) -> float:
        closes = self.candle_agg.get_closes()
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = float(np.mean(gains[-period:]))
        avg_l  = float(np.mean(losses[-period:]))
        if avg_l == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + avg_g / avg_l)

    def _atr(self, period: int = 14) -> float:
        closes = self.candle_agg.get_closes()
        highs  = self.candle_agg.get_highs()
        lows   = self.candle_agg.get_lows()
        if len(closes) < period + 1:
            return 0.0
        prev_c = closes[:-1]
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - prev_c), np.abs(lows[1:] - prev_c)),
        )
        return float(np.mean(tr[-period:]))

    # ── signal ───────────────────────────────────────────────
    def get_signal(self) -> str | None:
        if len(self.candle_agg.candles) < SMA_PERIOD:
            return None
        # Fee-drag filter: skip if volatility < round-trip fee cost
        if self.current_price > 0 and self.last_atr > 0:
            if (self.last_atr / self.current_price) < (0.00055 * 2):
                return None
        if self.last_rsi < RSI_BUY_THRESH and self.current_price > self.last_sma and self.last_imbalance > 0.55:
            return "Buy"
        if self.last_rsi > RSI_SELL_THRESH and self.current_price < self.last_sma and self.last_imbalance < 0.45:
            return "Sell"
        return None

    # ── TP / SL ──────────────────────────────────────────────
    def calculate_tp_sl(self, side: str, entry_price: float) -> tuple[float, float]:
        min_dist    = self.tick_size * MIN_TICK_DIST
        spread_dist = self.best_ask - self.best_bid
        sl_dist     = max(self.last_atr * SL_ATR_MULT, min_dist, spread_dist * 1.5)
        tp_dist     = max(self.last_atr * TP_ATR_MULT, min_dist)

        if side == "Buy":
            tp_raw = entry_price + tp_dist
            sl_raw = entry_price - sl_dist
        else:
            tp_raw = entry_price - tp_dist
            sl_raw = entry_price + sl_dist

        tp = round(round(tp_raw, 10) / self.tick_size) * self.tick_size
        sl = round(round(sl_raw, 10) / self.tick_size) * self.tick_size
        return tp, sl

    def sl_is_valid(self, side: str, sl_price: float) -> bool:
        """Reject SL that sits inside the current spread."""
        if side == "Sell" and sl_price <= self.best_ask:
            return False
        if side == "Buy" and sl_price >= self.best_bid:
            return False
        return True


# ════════════════════════════════════════════════════════════
#  MULTI-POSITION PAPER STATE
# ════════════════════════════════════════════════════════════
class MultiPositionState:
    """Shared balance, multiple concurrent positions, trade history."""

    FEE_RATE = 0.00055

    def __init__(self, initial_balance: float = PAPER_INITIAL_BAL):
        self.initial_balance = initial_balance
        self.balance         = initial_balance
        self.positions: dict[str, dict] = {}
        self.trades: list[dict]         = []
        self.total_pnl    = 0.0
        self.win_count    = 0
        self.loss_count   = 0
        self.gross_profit = 0.0
        self.gross_loss   = 0.0
        self.peak_equity  = initial_balance
        self.max_drawdown = 0.0

    # ── persistence ──────────────────────────────────────────
    def load_state(self):
        if not TRADE_LOG.exists():
            return
        with open(TRADE_LOG, "r") as fh:
            for line in fh:
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.trades.append(t)
                self.balance = t.get("balance_after", self.balance)
                pnl = t.get("pnl", 0.0)
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1; self.gross_profit += pnl
                else:
                    self.loss_count += 1; self.gross_loss += abs(pnl)
        self.peak_equity = max(self.peak_equity, self.balance)
        logger.info(f"📂 Loaded {len(self.trades)} trades. Balance: ${self.balance:.2f}")

    def _save_trade(self, rec: dict):
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        with open(TRADE_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    # ── position lifecycle ───────────────────────────────────
    def open_position(self, symbol: str, side: str, qty: float,
                      entry: float, tp: float, sl: float, tick_size: float):
        if symbol in self.positions:
            logger.warning(f"⚠️ [{symbol}] Position already open – closing first")
            self.close_position(symbol, entry, tick_size)

        self.positions[symbol] = {
            "side": side, "quantity": qty, "entry_price": entry,
            "tp_price": tp, "sl_price": sl, "tick_size": tick_size,
            "current_price": entry,
            "open_time": datetime.now(timezone.utc).isoformat(),
            "status": "open",
        }
        logger.info(
            f"📊 [{symbol}] Opened {side}: {qty:.4f} @ {entry:.8f} "
            f"| TP: {tp:.8f} | SL: {sl:.8f}"
        )

    def close_position(self, symbol: str, exit_price: float, tick_size: float) -> float:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return 0.0

        qty   = pos["quantity"]
        entry = pos["entry_price"]
        side  = pos["side"]

        raw_pnl = (exit_price - entry) * qty if side == "Buy" else (entry - exit_price) * qty
        pnl     = raw_pnl * LEVERAGE

        notional = qty * entry
        fees     = notional * self.FEE_RATE * 2 * LEVERAGE
        pnl     -= fees

        self.balance   += pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.win_count += 1; self.gross_profit += pnl
        else:
            self.loss_count += 1; self.gross_loss += abs(pnl)

        equity = self.get_equity()
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        margin     = notional / LEVERAGE
        pnl_pct    = (pnl / margin) * 100 if margin > 0 else 0.0

        rec = {
            **pos, "symbol": symbol,
            "exit_price": exit_price,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "pnl": pnl, "pnl_percent": pnl_pct,
            "balance_after": self.balance, "status": "closed",
        }
        self.trades.append(rec)
        self._save_trade(rec)

        logger.info(
            f"📊 [{symbol}] Closed {side} @ {exit_price:.8f} "
            f"| P&L: ${pnl:.4f} | Bal: ${self.balance:.2f}"
        )
        return pnl

    # ── TP / SL check (called from WS thread) ───────────────
    def check_tp_sl(self, symbol: str, best_bid: float, best_ask: float) -> float | None:
        """Returns pnl if a position was closed, else None."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        tick  = pos.get("tick_size", 0.000001)
        side  = pos["side"]
        tp    = pos["tp_price"]
        sl    = pos["sl_price"]

        # Realistic taker exit price
        exit_price = best_bid if side == "Buy" else best_ask

        hit = False
        if side == "Buy":
            if exit_price >= tp:
                hit = True
            elif exit_price <= sl:
                hit = True
                # Cap slippage
                worst = sl - tick * MAX_SLIP_TICKS
                exit_price = max(exit_price, worst)
        else:
            if exit_price <= tp:
                hit = True
            elif exit_price >= sl:
                hit = True
                worst = sl + tick * MAX_SLIP_TICKS
                exit_price = min(exit_price, worst)

        if hit:
            label = "TP" if (exit_price >= tp if side == "Buy" else exit_price <= tp) else "SL"
            emoji = "🎯" if label == "TP" else "🛑"
            logger.info(f"{emoji} [{symbol}] {label} hit!")
            return self.close_position(symbol, exit_price, tick)
        return None

    # ── equity / stats ───────────────────────────────────────
    def update_current_price(self, symbol: str, price: float):
        if symbol in self.positions:
            self.positions[symbol]["current_price"] = price

    def get_equity(self) -> float:
        eq = self.balance
        for pos in self.positions.values():
            cp    = pos.get("current_price", pos["entry_price"])
            entry = pos["entry_price"]
            qty   = pos["quantity"]
            if pos["side"] == "Buy":
                eq += (cp - entry) * qty * LEVERAGE
            else:
                eq += (entry - cp) * qty * LEVERAGE
        return eq

    def get_stats(self) -> dict:
        n = len(self.trades)
        if n == 0:
            return {
                "total_trades": 0, "win_rate": 0, "total_pnl": 0,
                "balance": self.balance, "equity": self.get_equity(),
                "max_drawdown": 0, "profit_factor": 0, "expectancy": 0,
                "open_positions": len(self.positions),
            }
        pf = self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf")
        return {
            "total_trades": n,
            "win_rate": (self.win_count / n) * 100,
            "total_pnl": self.total_pnl,
            "balance": self.balance,
            "equity": self.get_equity(),
            "max_drawdown": self.max_drawdown * 100,
            "profit_factor": pf,
            "expectancy": self.total_pnl / n,
            "open_positions": len(self.positions),
        }


# ════════════════════════════════════════════════════════════
#  MULTI-SYMBOL BOT
# ════════════════════════════════════════════════════════════
class MultiSymbolBot:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self.api_key    = os.getenv("BYBIT_API_KEY", "")
        self.api_secret = os.getenv("BYBIT_SECRET_KEY", "")
        if not self.api_key or not self.api_secret:
            raise ValueError("❌ API keys not found in .env")

        self.loop    = loop or asyncio.get_event_loop()
        self.session = HTTP(testnet=False, api_key=self.api_key, api_secret=self.api_secret)

        # Per-symbol trackers
        self.trackers: dict[str, SymbolTracker] = {}
        for sym in SYMBOLS:
            try:
                info = self.session.get_instruments_info(category="linear", symbol=sym)
                inst = info["result"]["list"][0]
                tick = float(inst["priceFilter"]["tickSize"])
                step = inst["lotSizeFilter"]["qtyStep"]
                self.trackers[sym] = SymbolTracker(sym, tick, step)
                logger.info(f"✅ [{sym}] tick={tick} | qty_step={step}")
            except Exception as exc:
                logger.error(f"❌ Failed to init {sym}: {exc}")

        if not self.trackers:
            raise RuntimeError("No valid symbols initialised")

        # State
        self.state = MultiPositionState()
        self.state.load_state()

        # Trading guards
        self.symbol_cooldowns: dict[str, float] = {}   # symbol → resume timestamp
        self.loss_streak       = 0
        self.cooldown_until    = 0.0
        self.last_trade_time   = 0.0
        self.is_trading        = False
        self.running           = False
        self._update_scheduled = False
        self._indicator_tick   = 0

        self.ws = None

    # ── WebSocket callback (background thread) ───────────────
    def handle_orderbook(self, message: dict):
        try:
            topic  = message.get("topic", "")
            symbol = topic.replace("orderbook.50.", "")
            tracker = self.trackers.get(symbol)
            if tracker is None:
                return

            data = message.get("data", {})
            tracker.update_orderbook(data.get("b", []), data.get("a", []))

            # Update current price for unrealized PnL
            self.state.update_current_price(symbol, tracker.current_price)

            # Check TP / SL
            pnl = self.state.check_tp_sl(symbol, tracker.best_bid, tracker.best_ask)
            if pnl is not None:
                if pnl < 0:
                    self.loss_streak += 1
                    self.symbol_cooldowns[symbol] = time.time() + SYMBOL_COOLDOWN
                    logger.info(f"⏳ [{symbol}] Cooldown {SYMBOL_COOLDOWN}s after loss")
                else:
                    self.loss_streak = 0

            # Throttle async logic
            if not self._update_scheduled:
                self._update_scheduled = True

                async def _wrapper():
                    try:
                        await self.on_tick()
                    finally:
                        self._update_scheduled = False

                asyncio.run_coroutine_threadsafe(_wrapper(), self.loop)

        except Exception as exc:
            logger.error(f"Orderbook handler error: {exc}")

    # ── Main trading logic (asyncio loop) ────────────────────
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

            # Refresh indicators every 10 ticks
            self._indicator_tick += 1
            if self._indicator_tick % 10 == 0:
                for t in self.trackers.values():
                    t.refresh_indicators()

            # Balance guard
            if self.state.balance < MIN_BALANCE:
                logger.critical("💀 Balance too low. Halting.")
                self.running = False
                return

            # Global loss-streak cooldown
            if self.loss_streak >= LOSS_STREAK_LIMIT:
                logger.warning(f"⛔ {self.loss_streak} consecutive losses. Cooling down 10 min.")
                self.cooldown_until = now + 600
                self.loss_streak    = 0
                self.is_trading     = False
                return

            # Scan symbols
            for symbol, tracker in self.trackers.items():
                if symbol in self.state.positions:
                    continue
                if len(self.state.positions) >= MAX_CONCURRENT:
                    break
                if symbol in self.symbol_cooldowns and now < self.symbol_cooldowns[symbol]:
                    continue
                if tracker.spread > MAX_SPREAD:
                    continue
                if tracker.best_bid <= 0 or tracker.best_ask <= 0:
                    continue

                signal = tracker.get_signal()
                if signal is None:
                    continue

                entry_price = tracker.best_ask if signal == "Buy" else tracker.best_bid
                tp_price, sl_price = tracker.calculate_tp_sl(signal, entry_price)

                # ★ Spread guard: reject if SL is inside the spread
                if not tracker.sl_is_valid(signal, sl_price):
                    logger.warning(
                        f"⛔ [{symbol}] Skipping {signal}: SL {sl_price} inside spread "
                        f"(bid={tracker.best_bid}, ask={tracker.best_ask})"
                    )
                    continue

                # Lot sizing
                notional = MAX_TRADE_SIZE * LEVERAGE
                qty_raw  = notional / entry_price
                qty_dec  = Decimal(str(qty_raw)).quantize(
                    Decimal(str(tracker.qty_step)), rounding=ROUND_DOWN
                )
                quantity = float(qty_dec)
                if quantity <= 0:
                    continue

                self._log_signal(symbol, tracker, signal)
                self.state.open_position(
                    symbol, signal, quantity, entry_price,
                    tp_price, sl_price, tracker.tick_size,
                )
                self.last_trade_time = time.time()
                self.loss_streak     = 0
                logger.success(f"✅ [{symbol}] {signal} executed!")
                break  # one trade per tick

            self.is_trading = False

        except Exception as exc:
            logger.error(f"Trading logic error: {exc}")
            self.is_trading = False

    # ── Signal logging ───────────────────────────────────────
    def _log_signal(self, symbol: str, tracker: SymbolTracker, signal: str):
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        exists = SIGNAL_LOG.exists()
        with open(SIGNAL_LOG, "a", newline="") as fh:
            w = csv.writer(fh)
            if not exists:
                w.writerow(["timestamp", "symbol", "signal", "price",
                            "rsi", "atr", "sma", "imbalance", "spread"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(), symbol, signal,
                tracker.current_price, f"{tracker.last_rsi:.2f}",
                f"{tracker.last_atr:.8f}", f"{tracker.last_sma:.8f}",
                f"{tracker.last_imbalance:.4f}", f"{tracker.spread:.6f}",
            ])

    # ── Stats persistence ────────────────────────────────────
    def save_stats(self):
        stats  = self.state.get_stats()
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        exists = PNL_LOG.exists()
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "balance": stats["balance"], "equity": stats["equity"],
            "total_pnl": stats["total_pnl"], "win_rate": stats["win_rate"],
            "total_trades": stats["total_trades"],
            "open_positions": stats["open_positions"],
            "max_drawdown": stats["max_drawdown"],
            "profit_factor": stats["profit_factor"],
        }
        with open(PNL_LOG, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=row.keys())
            if not exists:
                w.writeheader()
            w.writerow(row)

    # ── WebSocket setup ──────────────────────────────────────
    def start_websocket(self):
        try:
            self.ws = WebSocket(
                testnet=False, channel_type="linear",
                api_key=self.api_key, api_secret=self.api_secret,
            )
            for sym in self.trackers:
                self.ws.orderbook_stream(50, symbol=sym, callback=self.handle_orderbook)
                logger.info(f"🔌 Subscribed to {sym}")
            logger.info("✅ All WebSocket streams active!")
        except Exception as exc:
            logger.error(f"WebSocket init failed: {exc}")
            self.running = False

    # ── Main loop ────────────────────────────────────────────
    async def run(self):
        logger.info(f"🚀 Multi-Symbol Bot: {', '.join(self.trackers)}")
        logger.info(f"   Leverage: {LEVERAGE}x | Max Positions: {MAX_CONCURRENT}")
        logger.info(f"   Balance: ${self.state.balance:.2f}")

        self.running = True
        threading.Thread(target=self.start_websocket, daemon=True).start()

        while self.running:
            await asyncio.sleep(1)
            if int(time.time()) % 60 == 0:
                stats = self.state.get_stats()
                open_syms = list(self.state.positions.keys())
                logger.info(
                    f"💰 Bal: ${stats['balance']:.2f} | Eq: ${stats['equity']:.2f} | "
                    f"P&L: ${stats['total_pnl']:.2f} | Trades: {stats['total_trades']} | "
                    f"Open: {open_syms}"
                )
                self.save_stats()
                if self.state.balance < MIN_BALANCE:
                    self.running = False
                    break

        self._print_final()

    def _print_final(self):
        s = self.state.get_stats()
        logger.info("=" * 60)
        logger.info("📊 FINAL STATISTICS")
        logger.info(f"   Trades: {s['total_trades']} | Win Rate: {s['win_rate']:.1f}%")
        logger.info(f"   P&L: ${s['total_pnl']:.2f} | Balance: ${s['balance']:.2f}")
        logger.info(f"   Max DD: {s['max_drawdown']:.2f}% | PF: {s['profit_factor']:.2f}")
        logger.info("=" * 60)


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
async def main():
    bot = MultiSymbolBot(loop=asyncio.get_running_loop())
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user.")
    except Exception as exc:
        logger.error(f"❌ Fatal: {exc}")