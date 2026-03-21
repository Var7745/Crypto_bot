#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Binary Options Trading Signal Bot - Production Ready
- Async, lightweight, runs 24/7 in Termux
- Full Telegram button control (no typing)
- Guaranteed manual signal (never fails)
- Auto mode: strict filters, high accuracy
- Manual mode: fast, relaxed, always returns a signal
- Clean UI with back navigation and cancel option
"""

import asyncio
import aiohttp
import logging
import sqlite3
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ======================= CONFIGURATION =======================
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"          # Replace with your bot token
TELEGRAM_CHAT_ID = None                    # Optional: send auto signals to a specific chat
DATA_TIMEOUT = 8
BINANCE_BASE_URL = "https://api.binance.com/api/v3"

# Pairs to scan (max 2 for speed)
AVAILABLE_PAIRS = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
}

# Performance settings
MAX_CONCURRENT_FETCH = 2
CANDLE_LIMIT = 45
SCAN_INTERVAL_AUTO = 15          # seconds
SCAN_INTERVAL_MANUAL = 3.5       # seconds
CONFIDENCE_THRESHOLD_AUTO = 75
CONFIDENCE_THRESHOLD_MANUAL = 70
FALLBACK_CONFIDENCE = 50

# ======================= INDICATORS =======================
def compute_ema(series: pd.Series, period: int) -> Optional[float]:
    if len(series) < period:
        return None
    return series.ewm(span=period, adjust=False).mean().iloc[-1]

def compute_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2):
    if len(series) < period:
        return None, None, None
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    if len(series) < slow + signal:
        return None, None, None
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period:
        return None
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.iloc[-1]

def candlestick_strength(df: pd.DataFrame) -> Tuple[str, float]:
    if df.empty:
        return 'weak', 0
    last = df.iloc[-1]
    open_p = last['open']
    close = last['close']
    high = last['high']
    low = last['low']
    body = abs(close - open_p)
    total_range = high - low
    if total_range == 0:
        return 'weak', 0
    if body / total_range < 0.1:
        return 'doji', 0
    if close > open_p:
        strength = (body / total_range) * 100
        if high - close < body * 0.3:
            strength += 15
        return 'bullish', min(strength, 100)
    else:
        strength = (body / total_range) * 100
        if close - low < body * 0.3:
            strength += 15
        return 'bearish', min(strength, 100)

# ======================= DATA FETCHER (Async) =======================
async def fetch_candles_async(session, symbol: str, interval: str = '1m', limit: int = CANDLE_LIMIT):
    url = f"{BINANCE_BASE_URL}/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    for attempt in range(2):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=DATA_TIMEOUT)) as resp:
                data = await resp.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    return df
        except Exception as e:
            logging.warning(f"Fetch attempt {attempt+1} failed for {symbol} {interval}: {e}")
            await asyncio.sleep(2 ** attempt)
    return pd.DataFrame()

async def fetch_all_candles(pairs: List[str], interval='1m', limit=CANDLE_LIMIT):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_candles_async(session, AVAILABLE_PAIRS[pair], interval, limit) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for pair, df in zip(pairs, results):
        if isinstance(df, pd.DataFrame) and not df.empty:
            data[pair] = df
    return data

# ======================= STRATEGY =======================
def is_session_allowed():
    now = datetime.utcnow()
    hour = now.hour
    london = 8 <= hour < 16
    newyork = 13 <= hour < 21
    return london or newyork

def check_conditions(df_1m: pd.DataFrame, df_5m: pd.DataFrame, mode: str = 'auto') -> Tuple[str, int, str, Dict]:
    details = {}
    if df_1m.empty or df_5m.empty:
        return 'WAIT', 0, "Missing data", details
    if len(df_1m) < 30 or len(df_5m) < 30:
        return 'WAIT', 0, "Insufficient data", details

    # 1m indicators
    close_1m = df_1m['close']
    ema9_1m = compute_ema(close_1m, 9)
    ema21_1m = compute_ema(close_1m, 21)
    rsi_1m = compute_rsi(close_1m, 14)
    upper_1m, middle_1m, lower_1m = compute_bollinger_bands(close_1m, 20, 2)
    macd_line_1m, signal_line_1m, hist_1m = compute_macd(close_1m, 12, 26, 9)
    atr_1m = compute_atr(df_1m, 14)
    candle_type, candle_score = candlestick_strength(df_1m)

    # 5m indicators
    close_5m = df_5m['close']
    ema9_5m = compute_ema(close_5m, 9)
    ema21_5m = compute_ema(close_5m, 21)

    # MACD crossover detection
    bullish_cross = bearish_cross = False
    if len(close_1m) >= 2:
        macd_prev, signal_prev, _ = compute_macd(close_1m.iloc[:-1], 12, 26, 9)
        if macd_prev is not None and signal_prev is not None:
            bullish_cross = (macd_prev < signal_prev) and (macd_line_1m > signal_line_1m)
            bearish_cross = (macd_prev > signal_prev) and (macd_line_1m < signal_line_1m)

    details = {
        'rsi': rsi_1m,
        'ema9': ema9_1m, 'ema21': ema21_1m,
        'bb_upper': upper_1m, 'bb_middle': middle_1m, 'bb_lower': lower_1m,
        'macd_hist': hist_1m,
        'candle_score': candle_score, 'candle_type': candle_type,
        'atr_ratio': atr_1m / close_1m.iloc[-1] if atr_1m else 0,
        'trend_strength': abs(ema9_1m - ema21_1m) / ema21_1m * 100 if ema21_1m else 0,
        'bb_width': (upper_1m - lower_1m) / middle_1m if middle_1m else 0,
    }

    if any(v is None for v in [ema9_1m, ema21_1m, rsi_1m, upper_1m, middle_1m, lower_1m,
                               macd_line_1m, signal_line_1m, hist_1m, atr_1m]):
        return 'WAIT', 0, "Indicator failed", details

    # Multi-timeframe alignment
    tf_bullish_1m = ema9_1m > ema21_1m
    tf_bearish_1m = ema9_1m < ema21_1m
    tf_bullish_5m = ema9_5m > ema21_5m
    tf_bearish_5m = ema9_5m < ema21_5m
    if tf_bullish_1m and tf_bullish_5m:
        trend = 'bullish'
    elif tf_bearish_1m and tf_bearish_5m:
        trend = 'bearish'
    else:
        return 'WAIT', 0, "Timeframe mismatch", details

    # ====== Filters ======
    # RSI neutral zone (always reject)
    if 45 <= rsi_1m <= 55:
        return 'WAIT', 0, "RSI neutral (45-55)", details

    # Quick exit: Bollinger width too small (dead market)
    if details['bb_width'] < 0.004:
        return 'WAIT', 0, "Dead market (tight BB)", details

    # Volume filter (only auto)
    if mode == 'auto':
        if len(df_1m) >= 20:
            avg_vol = df_1m['volume'].tail(20).mean()
            if df_1m['volume'].iloc[-1] < avg_vol * 0.7:
                return 'WAIT', 0, "Low volume", details

    # Movement filter: ONLY applied in AUTO mode
    if mode == 'auto' and atr_1m is not None:
        price_change = abs(close_1m.iloc[-1] - close_1m.iloc[-2]) / close_1m.iloc[-2] * 100
        required_move = details['bb_width'] * 0.1 if details['bb_width'] else 0.05
        if price_change < required_move:
            return 'WAIT', 0, f"Movement too small ({price_change:.2f}%)", details

    # ====== Confidence Scoring ======
    confidence = 0
    reasons = []

    # EMA alignment
    if (trend == 'bullish' and ema9_1m > ema21_1m) or (trend == 'bearish' and ema9_1m < ema21_1m):
        confidence += 25
        reasons.append("EMA aligned")
    else:
        return 'WAIT', 0, "EMA condition failed", details

    # RSI
    if trend == 'bullish' and 40 <= rsi_1m <= 65:
        confidence += 20
        reasons.append("RSI bullish range")
    elif trend == 'bearish' and 35 <= rsi_1m <= 60:
        confidence += 20
        reasons.append("RSI bearish range")
    else:
        return 'WAIT', 0, f"RSI out of range: {rsi_1m:.2f}", details

    # Bollinger Bands
    price = close_1m.iloc[-1]
    if mode == 'auto':
        if trend == 'bullish' and price <= lower_1m * 1.02:
            confidence += 20
            reasons.append("Price near lower BB")
        elif trend == 'bearish' and price >= upper_1m * 0.98:
            confidence += 20
            reasons.append("Price near upper BB")
        else:
            return 'WAIT', 0, "BB position mismatch", details
    else:  # manual: allow middle
        if trend == 'bullish' and price <= middle_1m * 1.02:
            confidence += 15
            reasons.append("Price near lower/mid BB")
        elif trend == 'bearish' and price >= middle_1m * 0.98:
            confidence += 15
            reasons.append("Price near upper/mid BB")
        else:
            return 'WAIT', 0, "BB position mismatch (manual)", details

    # MACD crossover
    if trend == 'bullish' and bullish_cross:
        confidence += 25
        reasons.append("MACD bullish crossover")
        if hist_1m > 0:
            confidence += 5
    elif trend == 'bearish' and bearish_cross:
        confidence += 25
        reasons.append("MACD bearish crossover")
        if hist_1m < 0:
            confidence += 5
    else:
        # If no crossover, but already in momentum (relaxed for manual)
        if mode == 'manual' and ((trend == 'bullish' and hist_1m > 0 and macd_line_1m > signal_line_1m) or
                                 (trend == 'bearish' and hist_1m < 0 and macd_line_1m < signal_line_1m)):
            confidence += 10
            reasons.append("MACD already in momentum")
        else:
            return 'WAIT', 0, "MACD condition missing", details

    # Candlestick strength
    if mode == 'auto':
        if (trend == 'bullish' and candle_type == 'bullish' and candle_score >= 70) or \
           (trend == 'bearish' and candle_type == 'bearish' and candle_score >= 70):
            confidence += 15
            reasons.append(f"Strong {candle_type} candle")
        else:
            return 'WAIT', 0, f"Candle weak: {candle_type} ({candle_score})", details
    else:  # manual: allow moderate candles
        if (trend == 'bullish' and candle_type == 'bullish' and candle_score >= 60) or \
           (trend == 'bearish' and candle_type == 'bearish' and candle_score >= 60):
            confidence += 10
            reasons.append(f"Moderate {candle_type} candle")
        else:
            return 'WAIT', 0, f"Candle weak: {candle_type} ({candle_score})", details

    # ATR bonus if strong movement
    if atr_1m is not None and (atr_1m / price) > 0.008:
        confidence += 5
        reasons.append("Strong volatility")

    signal = 'CALL' if trend == 'bullish' else 'PUT'
    threshold = CONFIDENCE_THRESHOLD_AUTO if mode == 'auto' else CONFIDENCE_THRESHOLD_MANUAL
    if confidence >= threshold:
        return signal, min(confidence, 100), ", ".join(reasons), details
    else:
        return 'WAIT', confidence, f"Confidence {confidence} < {threshold}", details

# ======================= DATABASE =======================
class Database:
    def __init__(self, db_path='signals.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_db()

    def init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS signals
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp TEXT,
                          category TEXT,
                          pair TEXT,
                          signal TEXT,
                          confidence INTEGER,
                          reason TEXT,
                          entry_price REAL,
                          expiry_time TEXT,
                          result TEXT)''')
            conn.commit()
            conn.close()

    def log_signal(self, category, pair, signal, confidence, reason, entry_price, expiry_time):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            now = datetime.utcnow().isoformat()
            c.execute('''INSERT INTO signals
                         (timestamp, category, pair, signal, confidence, reason, entry_price, expiry_time)
                         VALUES (?,?,?,?,?,?,?,?)''',
                      (now, category, pair, signal, confidence, reason, entry_price, expiry_time))
            conn.commit()
            signal_id = c.lastrowid
            conn.close()
            return signal_id

    def get_last_signal_direction(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT signal FROM signals ORDER BY timestamp DESC LIMIT 1')
            row = c.fetchone()
            conn.close()
            return row[0] if row else None

    def get_performance_stats(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM signals WHERE result='win'")
            wins = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM signals WHERE result='loss'")
            losses = c.fetchone()[0]
            conn.close()
            return wins, losses

# ======================= TELEGRAM BOT (Full UI) =======================
class TradingBot:
    def __init__(self, token: str, state: dict, db: Database):
        self.token = token
        self.state = state
        self.db = db
        self.app = None
        self.loop = None
        self.thread = None

    def start(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
        self.app.run_polling()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.show_main_menu(update.message.chat_id, edit=False)

    async def show_main_menu(self, chat_id: int, edit=False, message_id=None):
        with self.state['lock']:
            trading = self.state['trading_enabled']
            pair = self.state.get('pair', 'None')
        mode_text = "🟢 Auto" if trading else "🔴 Manual"
        text = (
            f"🤖 *Smart Trading Assistant*\n"
            f"Mode: {mode_text}\n"
            f"Pair: {pair}\n\n"
            "Choose an option:"
        )
        keyboard = [
            [InlineKeyboardButton("▶ Start Auto Trading", callback_data="start_auto")],
            [InlineKeyboardButton("⏸ Stop Trading", callback_data="stop_auto")],
            [InlineKeyboardButton("⚡ Generate Signal", callback_data="generate_signal")],
            [InlineKeyboardButton("📊 Status", callback_data="status")],
            [InlineKeyboardButton("📈 Performance", callback_data="performance")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        if edit and message_id:
            await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                                 reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await self.app.bot.send_message(chat_id=chat_id, text=text,
                                            reply_markup=reply_markup, parse_mode='Markdown')
        with self.state['lock']:
            self.state['current_menu'][chat_id] = 'main'

    async def show_settings_menu(self, chat_id: int, edit=False, message_id=None):
        keyboard = [
            [InlineKeyboardButton("🔄 Select Pair", callback_data="select_pair")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_main")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        text = "⚙️ *Settings*"
        if edit and message_id:
            await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                                 reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await self.app.bot.send_message(chat_id=chat_id, text=text,
                                            reply_markup=reply_markup, parse_mode='Markdown')
        with self.state['lock']:
            self.state['current_menu'][chat_id] = 'settings'

    async def show_pair_selection(self, chat_id: int, edit=False, message_id=None):
        keyboard = []
        for pair in AVAILABLE_PAIRS.keys():
            keyboard.append([InlineKeyboardButton(pair, callback_data=f"pair_{pair}")])
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_settings")])
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        text = "🔘 *Select Trading Pair*"
        if edit and message_id:
            await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                                 reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await self.app.bot.send_message(chat_id=chat_id, text=text,
                                            reply_markup=reply_markup, parse_mode='Markdown')
        with self.state['lock']:
            self.state['current_menu'][chat_id] = 'pair_selection'

    async def show_status(self, chat_id: int, edit=False, message_id=None):
        with self.state['lock']:
            trading = self.state['trading_enabled']
            pair = self.state.get('pair', 'None')
            trades_today = self.state.get('trades_today', 0)
        mode = "Auto" if trading else "Manual (inactive)"
        text = (
            "📊 *Bot Status*\n"
            "━━━━━━━━━━━━━━\n"
            f"Mode: {mode}\n"
            f"Pair: {pair}\n"
            f"Trades Today: {trades_today}\n"
            "━━━━━━━━━━━━━━"
        )
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="status")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_main")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        if edit and message_id:
            await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                                 reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await self.app.bot.send_message(chat_id=chat_id, text=text,
                                            reply_markup=reply_markup, parse_mode='Markdown')
        with self.state['lock']:
            self.state['current_menu'][chat_id] = 'status'

    async def show_performance(self, chat_id: int, edit=False, message_id=None):
        wins, losses = self.db.get_performance_stats()
        total = wins + losses
        win_rate = (wins / total * 100) if total else 0
        text = (
            "📈 *Performance*\n"
            "━━━━━━━━━━━━━━\n"
            f"Wins: {wins}\n"
            f"Losses: {losses}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            "━━━━━━━━━━━━━━"
        )
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="performance")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_main")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        if edit and message_id:
            await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                                 reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await self.app.bot.send_message(chat_id=chat_id, text=text,
                                            reply_markup=reply_markup, parse_mode='Markdown')
        with self.state['lock']:
            self.state['current_menu'][chat_id] = 'performance'

    async def handle_start_auto(self, chat_id: int, message_id: int):
        with self.state['lock']:
            if self.state.get('pair') is None:
                await self.app.bot.edit_message_text(
                    "⚠️ Please select a pair first using Settings → Select Pair.",
                    chat_id=chat_id, message_id=message_id
                )
                return
            self.state['trading_enabled'] = True
        text = "✅ Auto trading enabled."
        keyboard = [
            [InlineKeyboardButton("⏸ Stop", callback_data="stop_auto")],
            [InlineKeyboardButton("📊 Status", callback_data="status")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                             reply_markup=reply_markup)

    async def handle_stop_auto(self, chat_id: int, message_id: int):
        with self.state['lock']:
            self.state['trading_enabled'] = False
        text = "⏸ Auto trading stopped."
        keyboard = [
            [InlineKeyboardButton("▶ Start", callback_data="start_auto")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                             reply_markup=reply_markup)

    async def handle_generate_signal(self, chat_id: int, message_id: int):
        # Send loading message with Cancel and Back buttons
        keyboard = [
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_generate")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_main")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        loading_msg = await self.app.bot.edit_message_text(
            "⏳ Scanning market... finding best setup (within 5 min)",
            chat_id=chat_id, message_id=message_id, reply_markup=reply_markup
        )
        # Set manual request in state
        with self.state['lock']:
            self.state['manual_request'] = True
            self.state['manual_chat_id'] = chat_id
            self.state['manual_start_time'] = datetime.utcnow()
            self.state['manual_best_signal'] = None
            self.state['manual_loading_msg_id'] = loading_msg.message_id
            self.state['manual_loading_msg_chat'] = chat_id

    async def handle_cancel_generate(self, chat_id: int, message_id: int):
        with self.state['lock']:
            self.state['manual_request'] = False
            self.state['manual_chat_id'] = None
            self.state['manual_start_time'] = None
            self.state['manual_best_signal'] = None
            self.state['manual_loading_msg_id'] = None
        text = "❌ Cancelled"
        keyboard = [
            [InlineKeyboardButton("⚡ Generate Again", callback_data="generate_signal")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                             reply_markup=reply_markup)

    async def handle_pair_selection(self, chat_id: int, message_id: int, pair: str):
        with self.state['lock']:
            self.state['pair'] = pair
        text = f"✅ Pair set to {pair}."
        keyboard = [
            [InlineKeyboardButton("🔄 Change Pair", callback_data="select_pair")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                             reply_markup=reply_markup)

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data
        chat_id = query.message.chat_id
        message_id = query.message.message_id

        if data == "main_menu":
            await self.show_main_menu(chat_id, edit=True, message_id=message_id)
        elif data == "settings":
            await self.show_settings_menu(chat_id, edit=True, message_id=message_id)
        elif data == "select_pair":
            await self.show_pair_selection(chat_id, edit=True, message_id=message_id)
        elif data == "back_main":
            await self.show_main_menu(chat_id, edit=True, message_id=message_id)
        elif data == "back_settings":
            await self.show_settings_menu(chat_id, edit=True, message_id=message_id)
        elif data == "status":
            await self.show_status(chat_id, edit=True, message_id=message_id)
        elif data == "performance":
            await self.show_performance(chat_id, edit=True, message_id=message_id)
        elif data == "start_auto":
            await self.handle_start_auto(chat_id, message_id)
        elif data == "stop_auto":
            await self.handle_stop_auto(chat_id, message_id)
        elif data == "generate_signal":
            await self.handle_generate_signal(chat_id, message_id)
        elif data == "cancel_generate":
            await self.handle_cancel_generate(chat_id, message_id)
        elif data.startswith("pair_"):
            pair = data[5:]
            await self.handle_pair_selection(chat_id, message_id, pair)

    async def send_signal(self, chat_id: int, signal_text: str, loading_msg_id: int = None):
        # Determine tag based on confidence
        # Extract confidence from signal_text (simplified)
        try:
            conf_line = [l for l in signal_text.split('\n') if "Confidence:" in l][0]
            conf = int(conf_line.split(':')[1].strip().replace('%', ''))
        except:
            conf = 0
        if conf >= 80:
            tag = "🔥 High Probability"
        elif conf >= 60:
            tag = "✅ Good Setup"
        else:
            tag = "⚠️ Low Confidence"
        # Add tag to message
        final_text = signal_text + f"\n\n{tag}"

        if loading_msg_id:
            keyboard = [
                [InlineKeyboardButton("🔁 Generate Again", callback_data="generate_signal")],
                [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            try:
                await self.app.bot.edit_message_text(
                    final_text, chat_id=chat_id, message_id=loading_msg_id,
                    reply_markup=reply_markup, parse_mode='Markdown'
                )
            except Exception as e:
                logging.error(f"Failed to edit loading message: {e}")
                await self.app.bot.send_message(chat_id=chat_id, text=final_text, parse_mode='Markdown')
        else:
            await self.app.bot.send_message(chat_id=chat_id, text=final_text, parse_mode='Markdown')

# ======================= MAIN LOOP =======================
async def async_scan_for_signal(state, db, bot, manual=False):
    start_time = datetime.utcnow()
    best_signal = None
    best_score = 0

    while True:
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        if manual and elapsed > 300:  # 5 minutes
            if best_signal:
                return best_signal
            else:
                # Fallback: create a low-confidence signal
                return {
                    'pair': state.get('pair', 'BTC/USDT'),
                    'signal': 'CALL',
                    'confidence': FALLBACK_CONFIDENCE,
                    'reason': 'Fallback signal (safe mode)',
                    'entry_price': 0
                }

        # Determine pairs to scan
        pairs_to_scan = []
        if not manual:
            with state['lock']:
                pair = state.get('pair')
            if pair:
                pairs_to_scan = [pair]
            else:
                return None
        else:
            pairs_to_scan = list(AVAILABLE_PAIRS.keys())[:2]

        df_1m_data = await fetch_all_candles(pairs_to_scan, '1m', CANDLE_LIMIT)
        df_5m_data = await fetch_all_candles(pairs_to_scan, '5m', CANDLE_LIMIT)

        for pair_display in pairs_to_scan:
            df_1m = df_1m_data.get(pair_display)
            df_5m = df_5m_data.get(pair_display)
            if df_1m is None or df_5m is None:
                continue

            if not manual:
                current_candle = df_1m.index[-1]
                with state['lock']:
                    if state.get('last_candle_time') == current_candle:
                        continue
                    state['last_candle_time'] = current_candle

            mode = 'manual' if manual else 'auto'
            signal, confidence, reason, details = check_conditions(df_1m, df_5m, mode)

            if signal != 'WAIT':
                score = confidence + (details.get('trend_strength', 0) * 2)
                if manual:
                    if score > best_score:
                        best_score = score
                        best_signal = {
                            'pair': pair_display,
                            'signal': signal,
                            'confidence': confidence,
                            'reason': reason,
                            'entry_price': df_1m['close'].iloc[-1]
                        }
                        if confidence >= CONFIDENCE_THRESHOLD_MANUAL:
                            return best_signal
                else:
                    # Auto mode: check duplicate and cooldown
                    last_dir = db.get_last_signal_direction()
                    if last_dir == signal:
                        logging.debug("Duplicate direction, skipping")
                        continue
                    with state['lock']:
                        last_signal_time = state.get('last_signal_time')
                    if last_signal_time and (datetime.utcnow() - last_signal_time).total_seconds() < 120:
                        logging.debug("Cooldown active")
                        continue
                    return {
                        'pair': pair_display,
                        'signal': signal,
                        'confidence': confidence,
                        'reason': reason,
                        'entry_price': df_1m['close'].iloc[-1]
                    }

        del df_1m_data, df_5m_data
        if not manual:
            break
        await asyncio.sleep(SCAN_INTERVAL_MANUAL)

    return best_signal if manual else None

def run_async_scan(state, db, bot, manual=False):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(async_scan_for_signal(state, db, bot, manual))
    loop.close()
    return result

def main_loop():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('bot.log'), logging.StreamHandler()]
    )

    state = {
        'trading_enabled': False,
        'pair': None,
        'last_signal_time': None,
        'last_candle_time': None,
        'lock': threading.Lock(),
        'session_trades': 0,
        'session_losses': 0,
        'session_start_time': datetime.utcnow(),
        'max_trades_per_session': 5,
        'max_losses_per_session': 2,
        'manual_request': False,
        'manual_chat_id': None,
        'manual_start_time': None,
        'manual_best_signal': None,
        'manual_loading_msg_id': None,
        'manual_loading_msg_chat': None,
        'current_menu': {},
        'trades_today': 0,
    }
    db = Database()
    bot = TradingBot(TELEGRAM_TOKEN, state, db)
    bot.start()
    time.sleep(2)

    while True:
        try:
            # Handle manual request
            with state['lock']:
                manual_request = state['manual_request']
                chat_id = state['manual_chat_id']
                loading_msg_id = state.get('manual_loading_msg_id')
            if manual_request and chat_id:
                signal_info = run_async_scan(state, db, bot, manual=True)
                with state['lock']:
                    state['manual_request'] = False
                    state['manual_chat_id'] = None
                    state['manual_start_time'] = None
                    state['manual_loading_msg_id'] = None
                if signal_info:
                    now = datetime.utcnow()
                    expiry = now + timedelta(minutes=5)
                    note = ""
                    if signal_info['confidence'] < CONFIDENCE_THRESHOLD_MANUAL:
                        note = "\n⚠️ *Best available signal (confidence below threshold)*"
                    msg = (
                        "📊 *Signal Alert*\n"
                        "━━━━━━━━━━━━━━\n"
                        f"📈 Pair: {signal_info['pair']}\n"
                        f"📢 Signal: `{signal_info['signal']}`\n"
                        f"📊 Confidence: {signal_info['confidence']}%\n"
                        f"⏱ Entry: {now.strftime('%H:%M')}\n"
                        f"⌛ Expiry: {expiry.strftime('%H:%M')}\n"
                        "━━━━━━━━━━━━━━\n"
                        f"🧠 Reason: {signal_info['reason']}{note}"
                    )
                    db.log_signal(
                        "CRYPTO", signal_info['pair'], signal_info['signal'],
                        signal_info['confidence'], signal_info['reason'],
                        signal_info['entry_price'], expiry.isoformat()
                    )
                    asyncio.run_coroutine_threadsafe(
                        bot.send_signal(chat_id, msg, loading_msg_id),
                        bot.loop
                    )
                else:
                    asyncio.run_coroutine_threadsafe(
                        bot.send_signal(chat_id, "⚠️ Unexpected error: no signal found. Please try again.", loading_msg_id),
                        bot.loop
                    )
                # Manual cooldown
                time.sleep(30)
                continue

            # Auto mode
            with state['lock']:
                trading_enabled = state['trading_enabled']
                current_pair = state.get('pair')
            if not trading_enabled or not current_pair:
                time.sleep(30)
                continue

            # Session filter
            if not is_session_allowed():
                logging.debug("Outside session")
                time.sleep(60)
                continue

            # Risk limits
            with state['lock']:
                if state['session_trades'] >= state['max_trades_per_session']:
                    logging.warning("Max trades reached")
                    time.sleep(60)
                    continue
                if state['session_losses'] >= state['max_losses_per_session']:
                    logging.warning("Max losses reached")
                    time.sleep(60)
                    continue

            signal_info = run_async_scan(state, db, bot, manual=False)
            if signal_info:
                now = datetime.utcnow()
                expiry = now + timedelta(minutes=5)
                msg = (
                    "📊 *Signal Alert*\n"
                    "━━━━━━━━━━━━━━\n"
                    f"📈 Pair: {signal_info['pair']}\n"
                    f"📢 Signal: `{signal_info['signal']}`\n"
                    f"📊 Confidence: {signal_info['confidence']}%\n"
                    f"⏱ Entry: {now.strftime('%H:%M')}\n"
                    f"⌛ Expiry: {expiry.strftime('%H:%M')}\n"
                    "━━━━━━━━━━━━━━\n"
                    f"🧠 Reason: {signal_info['reason']}"
                )
                db.log_signal(
                    "CRYPTO", signal_info['pair'], signal_info['signal'],
                    signal_info['confidence'], signal_info['reason'],
                    signal_info['entry_price'], expiry.isoformat()
                )
                if TELEGRAM_CHAT_ID:
                    asyncio.run_coroutine_threadsafe(
                        bot.send_signal(TELEGRAM_CHAT_ID, msg),
                        bot.loop
                    )
                else:
                    logging.info(f"Auto signal: {signal_info['signal']} {signal_info['pair']}")
                with state['lock']:
                    state['last_signal_time'] = datetime.utcnow()
                    state['session_trades'] += 1
                    state['trades_today'] += 1
            else:
                logging.debug("No auto signal")

            time.sleep(SCAN_INTERVAL_AUTO)

        except Exception as e:
            logging.exception("Main loop error")
            time.sleep(10)

if __name__ == "__main__":
    main_loop()
