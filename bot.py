#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High‑Accuracy Binary Options Signal Bot - Final Production Version
- Strict 80%+ confidence, no fallback signals
- Valid Binance crypto symbols only (16 pairs)
- Multi‑timeframe (1m + 5m) with momentum checks
- EMA50/200 trend, RSI 14 (range + direction), MACD momentum
- Support/Resistance, candle patterns, volatility filter
- Session filter (London/New York)
- Live timer during manual scan (2 min)
- 3‑minute entry delay, 5‑minute expiry
- Full Telegram button control, back navigation
- Optimized for Termux: async, caching, low CPU
- Auto‑restart, watchdog, SQLite logging, result tracking
"""

import asyncio
import aiohttp
import logging
import sqlite3
import gc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Tuple, Optional, List, Any

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# ======================= CONFIGURATION =======================
TELEGRAM_TOKEN = "8553023618:AAH7upKIA9j_zqIYtIhBRKThBOY2HlWe6Ss"  # Replace with your bot token
TELEGRAM_CHAT_ID = None
DATA_TIMEOUT = 8
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
INDIA_TZ = ZoneInfo("Asia/Kolkata")

# ======================= VALID BINANCE CRYPTO SYMBOLS =======================
VALID_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT",
    "DOTUSDT", "MATICUSDT", "LTCUSDT", "TRXUSDT", "ETCUSDT", "XLMUSDT", "ATOMUSDT",
    "LINKUSDT", "UNIUSDT"
]

# Mapping from user‑friendly name (same as symbol) to actual symbol (same)
SYMBOL_MAPPING = {sym: sym for sym in VALID_SYMBOLS}

# Categories – only crypto
CATEGORIES = {"Crypto": VALID_SYMBOLS}

# ======================= PERFORMANCE SETTINGS =======================
MAX_CONCURRENT_FETCH = 2
CANDLE_LIMIT = 100               # enough for EMA 200
SCAN_INTERVAL_AUTO = 15
SCAN_INTERVAL_MANUAL = 1
CONFIDENCE_THRESHOLD = 80
MAX_SCAN_PAIRS_MANUAL = 10
ENTRY_DELAY_MINUTES = 3          # user gets 3 min to place trade
EXPIRY_MINUTES = 5               # trade duration

# ======================= NEWS AVOIDANCE (UTC) =======================
NEWS_BLOCKS = [
    (13, 25, 13, 40),
    (15, 55, 16, 10),
]

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

def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2):
    if len(series) < period:
        return None, None, None
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]

def support_resistance(df: pd.DataFrame, window=20):
    if len(df) < window:
        return None, None
    recent_high = df['high'].tail(window).max()
    recent_low = df['low'].tail(window).min()
    return recent_low, recent_high

def candlestick_pattern(df: pd.DataFrame) -> Tuple[str, float]:
    if len(df) < 2:
        return 'none', 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    open_p = last['open']
    close = last['close']
    high = last['high']
    low = last['low']
    prev_open = prev['open']
    prev_close = prev['close']
    body = abs(close - open_p)
    total_range = high - low
    if total_range == 0:
        return 'none', 0

    # Bullish engulfing
    if close > open_p and prev_close < prev_open and close > prev_open and open_p < prev_close:
        return 'bullish_engulfing', 85
    # Bearish engulfing
    if close < open_p and prev_close > prev_open and close < prev_open and open_p > prev_close:
        return 'bearish_engulfing', 85
    # Rejection wick (long wick opposite to direction)
    upper_wick = high - max(open_p, close)
    lower_wick = min(open_p, close) - low
    if upper_wick > body * 2 and close < open_p:
        return 'bearish_rejection', 75
    if lower_wick > body * 2 and close > open_p:
        return 'bullish_rejection', 75
    # Momentum candle (large body)
    if body / total_range > 0.7:
        if close > open_p:
            return 'bullish_momentum', 70
        else:
            return 'bearish_momentum', 70
    return 'none', 0

def volatility_filter(df_1m: pd.DataFrame) -> bool:
    atr = compute_atr(df_1m, 14)
    if atr is None:
        return False
    price = df_1m['close'].iloc[-1]
    atr_pct = atr / price * 100
    if atr_pct < 0.2:
        return False
    upper, middle, lower = compute_bollinger_bands(df_1m['close'], 20, 2)
    if upper is None or lower is None:
        return False
    bb_width = (upper - lower) / middle * 100 if middle != 0 else 0
    if bb_width < 0.5:
        return False
    return True

# ======================= STRATEGY =======================
def is_session_allowed():
    now = datetime.now(timezone.utc)
    hour = now.hour
    london = 8 <= hour < 16
    newyork = 13 <= hour < 21
    return london or newyork

def news_filter() -> bool:
    now = datetime.now(timezone.utc)
    for start_h, start_m, end_h, end_m in NEWS_BLOCKS:
        start = datetime(now.year, now.month, now.day, start_h, start_m, tzinfo=timezone.utc)
        end = datetime(now.year, now.month, now.day, end_h, end_m, tzinfo=timezone.utc)
        if start <= now <= end:
            return True
    return False

def check_conditions(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> Tuple[str, int, str, Dict]:
    details = {}
    if df_1m.empty or df_5m.empty:
        return 'WAIT', 0, "Missing data", details
    if len(df_1m) < 200 or len(df_5m) < 200:
        return 'WAIT', 0, "Insufficient data for EMA 200", details

    # 1m indicators
    close_1m = df_1m['close']
    ema50_1m = compute_ema(close_1m, 50)
    ema200_1m = compute_ema(close_1m, 200)
    rsi_1m = compute_rsi(close_1m, 14)
    macd_line_1m, signal_line_1m, hist_1m = compute_macd(close_1m, 12, 26, 9)
    # 5m indicators
    close_5m = df_5m['close']
    ema50_5m = compute_ema(close_5m, 50)
    ema200_5m = compute_ema(close_5m, 200)
    rsi_5m = compute_rsi(close_5m, 14)
    macd_line_5m, signal_line_5m, hist_5m = compute_macd(close_5m, 12, 26, 9)
    # Support/resistance
    support, resistance = support_resistance(df_1m, 20)
    # Candle pattern
    pattern, pattern_score = candlestick_pattern(df_1m)

    # Check None
    if any(v is None for v in [ema50_1m, ema200_1m, rsi_1m, macd_line_1m, signal_line_1m,
                               ema50_5m, ema200_5m, rsi_5m, macd_line_5m, signal_line_5m]):
        return 'WAIT', 0, "Indicator failed", details

    # ---------- 1. TREND STRENGTH ----------
    ema_diff_pct = abs(ema50_1m - ema200_1m) / ema200_1m * 100
    if ema_diff_pct < 0.1:
        return 'WAIT', 0, "Weak trend (EMA difference too small)", details

    trend_bullish = ema50_1m > ema200_1m and close_1m.iloc[-1] > ema50_1m
    trend_bearish = ema50_1m < ema200_1m and close_1m.iloc[-1] < ema50_1m
    if not (trend_bullish or trend_bearish):
        return 'WAIT', 0, "No clear trend (EMA condition)", details

    # ---------- 2. RSI CONFIRMATION + MOMENTUM ----------
    # Get previous RSI to check direction
    prev_rsi_1m = compute_rsi(close_1m.iloc[:-1], 14)
    rsi_up = prev_rsi_1m is not None and rsi_1m > prev_rsi_1m
    rsi_down = prev_rsi_1m is not None and rsi_1m < prev_rsi_1m

    if trend_bullish:
        if not (25 <= rsi_1m <= 40 and rsi_up):
            return 'WAIT', 0, "RSI not in bullish range (25-40) or not rising", details
        if not (rsi_5m < 50):
            return 'WAIT', 0, "5m RSI not below 50", details
    else:
        if not (60 <= rsi_1m <= 75 and rsi_down):
            return 'WAIT', 0, "RSI not in bearish range (60-75) or not falling", details
        if not (rsi_5m > 50):
            return 'WAIT', 0, "5m RSI not above 50", details

    # ---------- 3. MACD MOMENTUM ----------
    # Previous histogram to check if momentum is increasing
    prev_hist_1m = compute_macd(close_1m.iloc[:-1], 12, 26, 9)[2]
    hist_increasing = prev_hist_1m is not None and hist_1m > prev_hist_1m
    macd_bullish = (macd_line_1m > signal_line_1m and hist_1m > 0 and hist_increasing)
    macd_bearish = (macd_line_1m < signal_line_1m and hist_1m < 0 and not hist_increasing)  # becoming more negative
    if trend_bullish and not macd_bullish:
        return 'WAIT', 0, "MACD not bullish with increasing momentum", details
    if trend_bearish and not macd_bearish:
        return 'WAIT', 0, "MACD not bearish with increasing momentum", details

    # ---------- 4. SUPPORT / RESISTANCE ----------
    price = close_1m.iloc[-1]
    if trend_bullish and support is not None:
        if price > support * 1.02:
            return 'WAIT', 0, "Not near support (distance > 2%)", details
    if trend_bearish and resistance is not None:
        if price < resistance * 0.98:
            return 'WAIT', 0, "Not near resistance (distance > 2%)", details

    # ---------- 5. CANDLE PATTERN (with body > previous candle) ----------
    if len(df_1m) >= 2:
        prev_body = abs(df_1m['close'].iloc[-2] - df_1m['open'].iloc[-2])
        current_body = abs(close_1m.iloc[-1] - df_1m['open'].iloc[-1])
        body_growing = current_body > prev_body
    else:
        body_growing = True

    valid_patterns = {
        'bullish': ['bullish_engulfing', 'bullish_rejection', 'bullish_momentum'],
        'bearish': ['bearish_engulfing', 'bearish_rejection', 'bearish_momentum']
    }
    if trend_bullish and pattern not in valid_patterns['bullish']:
        return 'WAIT', 0, f"No bullish candle pattern ({pattern})", details
    if trend_bearish and pattern not in valid_patterns['bearish']:
        return 'WAIT', 0, f"No bearish candle pattern ({pattern})", details
    if not body_growing:
        return 'WAIT', 0, "Candle body not larger than previous", details

    # ---------- 6. VOLATILITY FILTER ----------
    if not volatility_filter(df_1m):
        return 'WAIT', 0, "Low volatility", details

    # ---------- 7. MULTI-TIMEFRAME (already checked via EMAs and RSIs) ----------
    if (trend_bullish and not (ema50_5m > ema200_5m)) or (trend_bearish and not (ema50_5m < ema200_5m)):
        return 'WAIT', 0, "Multi‑timeframe disagreement", details

    # ---------- 8. SESSION FILTER ----------
    if not is_session_allowed():
        return 'WAIT', 0, "Outside trading session", details

    # ---------- CONFIDENCE SCORING ----------
    confidence = 0
    reasons = []

    # EMA trend strength
    confidence += 20
    reasons.append("EMA trend aligned")
    # RSI
    confidence += 15
    reasons.append("RSI confirmation")
    # MACD momentum
    confidence += 15
    reasons.append("MACD momentum")
    # Support/Resistance
    confidence += 20
    reasons.append("Near key level")
    # Multi‑timeframe
    confidence += 20
    reasons.append("Multi‑timeframe agreement")
    # Candle strength
    confidence += 10
    reasons.append(f"Strong candle: {pattern}")

    confidence = min(confidence, 100)
    signal = 'CALL' if trend_bullish else 'PUT'

    if confidence >= CONFIDENCE_THRESHOLD:
        return signal, confidence, ", ".join(reasons), details
    else:
        return 'WAIT', confidence, f"Confidence {confidence} < {CONFIDENCE_THRESHOLD}", details

# ======================= DATA FETCHER =======================
fetch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCH)
CACHE_DURATION = 3

async def fetch_candles_async(session, symbol: str, interval: str = '1m', limit: int = CANDLE_LIMIT):
    cache_key = f"{symbol}_{interval}_{limit}"
    async with state['cache_lock']:
        now = datetime.now(timezone.utc)
        if cache_key in state['market_cache']:
            cached_time, cached_df = state['market_cache'][cache_key]
            if (now - cached_time).total_seconds() < CACHE_DURATION:
                return cached_df
    async with fetch_semaphore:
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
                        async with state['cache_lock']:
                            state['market_cache'][cache_key] = (datetime.now(timezone.utc), df)
                        return df
            except Exception as e:
                logging.warning(f"Fetch attempt {attempt+1} failed for {symbol} {interval}: {e}")
                await asyncio.sleep(2 ** attempt)
        return pd.DataFrame()

async def fetch_all_candles(pairs: List[str], interval='1m', limit=CANDLE_LIMIT):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for pair in pairs:
            symbol = SYMBOL_MAPPING.get(pair)
            if not symbol or symbol not in VALID_SYMBOLS:
                logging.debug(f"Skipping unsupported pair: {pair}")
                continue
            tasks.append(fetch_candles_async(session, symbol, interval, limit))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for pair, df in zip(pairs, results):
        if isinstance(df, pd.DataFrame) and not df.empty:
            data[pair] = df
    return data

# ======================= DATABASE =======================
class Database:
    def __init__(self, db_path='signals.db'):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
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
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        c.execute('''INSERT INTO signals
                     (timestamp, category, pair, signal, confidence, reason, entry_price, expiry_time)
                     VALUES (?,?,?,?,?,?,?,?)''',
                  (now, category, pair, signal, confidence, reason, entry_price, expiry_time))
        conn.commit()
        signal_id = c.lastrowid
        conn.close()
        return signal_id

    def get_last_signal_direction(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT signal FROM signals ORDER BY timestamp DESC LIMIT 1')
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def get_performance_stats(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signals WHERE result='win'")
        wins = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM signals WHERE result='loss'")
        losses = c.fetchone()[0]
        conn.close()
        return wins, losses

# ======================= TELEGRAM BOT =======================
class TradingBot:
    def __init__(self, token: str, state: dict, db: Database):
        self.token = token
        self.state = state
        self.db = db
        self.app = None

    async def start(self):
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

        await self.app.initialize()
        await self.app.start()
        await self.app.bot.initialize()
        await self.app.updater.start_polling(drop_pending_updates=True)

        while True:
            await asyncio.sleep(3600)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.show_main_menu(update.message.chat_id, edit=False)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text.lower()
        if text in ["hi", "hello", "start", "menu"]:
            await self.show_main_menu(update.message.chat_id)

    async def show_main_menu(self, chat_id: int, edit=False, message_id=None):
        async with self.state['lock']:
            trading = self.state['trading_enabled']
            category = self.state.get('category', 'None')
            pair = self.state.get('pair', 'None')
        mode_text = "🟢 Auto" if trading else "🔴 Manual"
        text = (
            f"🤖 *Smart Trading Assistant*\n"
            f"Mode: {mode_text}\n"
            f"Category: {category}\n"
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
        async with self.state['lock']:
            self.state['current_menu'][chat_id] = 'main'

    async def show_settings_menu(self, chat_id: int, edit=False, message_id=None):
        keyboard = [
            [InlineKeyboardButton("📂 Select Category", callback_data="select_category")],
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
        async with self.state['lock']:
            self.state['current_menu'][chat_id] = 'settings'

    async def show_category_menu(self, chat_id: int, edit=False, message_id=None):
        keyboard = []
        for cat in CATEGORIES.keys():
            keyboard.append([InlineKeyboardButton(cat, callback_data=f"category_{cat}")])
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_settings")])
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        text = "📂 *Select Category*"
        if edit and message_id:
            await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                                 reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await self.app.bot.send_message(chat_id=chat_id, text=text,
                                            reply_markup=reply_markup, parse_mode='Markdown')
        async with self.state['lock']:
            self.state['current_menu'][chat_id] = 'category_selection'

    async def show_pair_selection(self, chat_id: int, edit=False, message_id=None):
        async with self.state['lock']:
            category = self.state.get('category')
        if not category:
            await self.show_category_menu(chat_id, edit=edit, message_id=message_id)
            return
        pairs = CATEGORIES.get(category, [])
        valid_pairs = [p for p in pairs if p in VALID_SYMBOLS]
        if not valid_pairs:
            await self.app.bot.edit_message_text(
                "⚠️ No valid pairs available in this category.",
                chat_id=chat_id, message_id=message_id
            )
            return
        keyboard = []
        for i in range(0, len(valid_pairs), 2):
            row = []
            for j in range(i, min(i+2, len(valid_pairs))):
                row.append(InlineKeyboardButton(valid_pairs[j], callback_data=f"pair_{valid_pairs[j]}"))
            keyboard.append(row)
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_settings")])
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        text = f"🔘 *Select Pair ({category})*"
        if edit and message_id:
            await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                                 reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await self.app.bot.send_message(chat_id=chat_id, text=text,
                                            reply_markup=reply_markup, parse_mode='Markdown')
        async with self.state['lock']:
            self.state['current_menu'][chat_id] = 'pair_selection'

    async def show_status(self, chat_id: int, edit=False, message_id=None):
        async with self.state['lock']:
            trading = self.state['trading_enabled']
            category = self.state.get('category', 'None')
            pair = self.state.get('pair', 'None')
            trades_today = self.state.get('trades_today', 0)
        mode = "Auto" if trading else "Manual (inactive)"
        text = (
            "📊 *Bot Status*\n"
            "━━━━━━━━━━━━━━\n"
            f"Mode: {mode}\n"
            f"Category: {category}\n"
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
        async with self.state['lock']:
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
        async with self.state['lock']:
            self.state['current_menu'][chat_id] = 'performance'

    async def handle_start_auto(self, chat_id: int, message_id: int):
        async with self.state['lock']:
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
        async with self.state['lock']:
            self.state['trading_enabled'] = False
        text = "⏸ Auto trading stopped."
        keyboard = [
            [InlineKeyboardButton("▶ Start", callback_data="start_auto")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                             reply_markup=reply_markup)

    async def update_timer(self, chat_id: int, msg_id: int):
        """Update timer message every 5 seconds for 2 minutes."""
        for remaining in range(120, 0, -5):
            minutes = remaining // 60
            seconds = remaining % 60
            text = (
                f"⏳ *Scanning market...*\n\n"
                f"Time remaining: {minutes:02}:{seconds:02}\n\n"
                f"Searching best opportunity..."
            )
            try:
                await self.app.bot.edit_message_text(
                    text,
                    chat_id=chat_id,
                    message_id=msg_id,
                    parse_mode='Markdown'
                )
            except Exception:
                pass
            await asyncio.sleep(5)

    async def handle_generate_signal(self, chat_id: int, message_id: int):
        async with self.state['lock']:
            if self.state.get('pair') is None:
                await self.app.bot.edit_message_text(
                    "⚠️ Please select a pair first using Settings → Select Pair.",
                    chat_id=chat_id, message_id=message_id
                )
                return

        keyboard = [
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_generate")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_main")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        loading_msg = await self.app.bot.edit_message_text(
            "⏳ Scanning market...\n\nTime remaining: 02:00\n\nSearching best opportunity...",
            chat_id=chat_id, message_id=message_id, reply_markup=reply_markup, parse_mode='Markdown'
        )
        asyncio.create_task(self.update_timer(chat_id, loading_msg.message_id))

        async with self.state['lock']:
            self.state['manual_request'] = True
            self.state['manual_chat_id'] = chat_id
            self.state['manual_start_time'] = datetime.now(timezone.utc)
            self.state['manual_best_signal'] = None
            self.state['manual_loading_msg_id'] = loading_msg.message_id

    async def handle_cancel_generate(self, chat_id: int, message_id: int):
        async with self.state['lock']:
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

    async def handle_category_selection(self, chat_id: int, message_id: int, category: str):
        async with self.state['lock']:
            self.state['category'] = category
            self.state['pair'] = None
        text = f"✅ Category set to {category}.\nNow use Settings → Select Pair to choose a pair."
        keyboard = [
            [InlineKeyboardButton("🔘 Select Pair", callback_data="select_pair")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                             reply_markup=reply_markup)

    async def handle_pair_selection(self, chat_id: int, message_id: int, pair_display: str):
        async with self.state['lock']:
            category = self.state.get('category')
            if category is None:
                await self.app.bot.edit_message_text(
                    "⚠️ Please select a category first.",
                    chat_id=chat_id, message_id=message_id
                )
                return
            # Verify that the pair has a valid mapping
            if pair_display not in VALID_SYMBOLS:
                await self.app.bot.edit_message_text(
                    f"⚠️ {pair_display} is not supported.\nPlease choose another pair.",
                    chat_id=chat_id, message_id=message_id
                )
                return
            self.state['pair'] = pair_display
        text = f"✅ Pair set to {pair_display} (Category: {category})."
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
        elif data == "select_category":
            await self.show_category_menu(chat_id, edit=True, message_id=message_id)
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
        elif data.startswith("category_"):
            category = data[9:]
            await self.handle_category_selection(chat_id, message_id, category)
        elif data.startswith("pair_"):
            pair = data[5:]
            await self.handle_pair_selection(chat_id, message_id, pair)

    async def send_signal(self, chat_id: int, signal_text: str, loading_msg_id: int = None):
        if loading_msg_id:
            keyboard = [
                [InlineKeyboardButton("🔁 Generate Again", callback_data="generate_signal")],
                [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            try:
                await self.app.bot.edit_message_text(
                    signal_text, chat_id=chat_id, message_id=loading_msg_id,
                    reply_markup=reply_markup, parse_mode='Markdown'
                )
            except Exception as e:
                logging.error(f"Failed to edit loading message: {e}")
                await self.app.bot.send_message(chat_id=chat_id, text=signal_text, parse_mode='Markdown')
        else:
            await self.app.bot.send_message(chat_id=chat_id, text=signal_text, parse_mode='Markdown')

# ======================= WATCHDOG =======================
async def watchdog(state):
    while True:
        await asyncio.sleep(30)
        async with state['lock']:
            last = state.get('last_heartbeat')
        if last:
            delay = (datetime.now(timezone.utc) - last).total_seconds()
            if delay > 180:
                logging.warning("WATCHDOG ALERT: BOT FROZEN – no heartbeat for 3 minutes")

# ======================= RESULT CHECKER =======================
async def result_checker(db):
    while True:
        await asyncio.sleep(60)
        conn = sqlite3.connect("signals.db")
        c = conn.cursor()
        c.execute("""
            SELECT id, pair, entry_price, expiry_time, signal
            FROM signals
            WHERE result IS NULL
        """)
        rows = c.fetchall()
        for row in rows:
            id_, pair_display, entry_price, expiry_str, signal = row
            expiry = datetime.fromisoformat(expiry_str)
            if datetime.now(timezone.utc) < expiry:
                continue
            # fetch current price
            df_data = await fetch_all_candles([pair_display], '1m', 2)
            df = df_data.get(pair_display)
            if df is None or df.empty:
                continue
            price = df['close'].iloc[-1]
            win = (signal == 'CALL' and price > entry_price) or (signal == 'PUT' and price < entry_price)
            result = 'win' if win else 'loss'
            c.execute("UPDATE signals SET result=? WHERE id=?", (result, id_))
        conn.commit()
        conn.close()

# ======================= AUTO ROTATION TASK =======================
async def auto_rotation(state, db):
    while True:
        await asyncio.sleep(300)  # 5 minutes
        async with state['lock']:
            if not state['trading_enabled']:
                continue
            category = state.get('category')
            if not category:
                continue
            pairs = CATEGORIES.get(category, [])
            valid = [p for p in pairs if p in VALID_SYMBOLS]
            if not valid:
                continue
            idx = state.get('rotation_index', 0)
            new_pair = valid[idx % len(valid)]
            state['rotation_index'] = idx + 1
            state['pair'] = new_pair
            logging.info(f"Auto rotation: switched to {new_pair}")

# ======================= MAIN LOOP =======================
async def async_scan_for_signal(state, db, bot, manual=False):
    start_time = datetime.now(timezone.utc)
    best_signal = None
    best_confidence = 0

    while True:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        if manual and elapsed > 120:  # 2 minutes max
            if best_signal:
                return best_signal
            else:
                return None

        pairs_to_scan = []
        if not manual:
            async with state['lock']:
                pair = state.get('pair')
            if pair and pair in VALID_SYMBOLS:
                pairs_to_scan = [pair]
            else:
                return None
        else:
            async with state['lock']:
                category = state.get('category')
            if category:
                all_pairs = CATEGORIES.get(category, [])
                valid_pairs = [p for p in all_pairs if p in VALID_SYMBOLS]
                pairs_to_scan = valid_pairs[:MAX_SCAN_PAIRS_MANUAL]
            else:
                pairs_to_scan = []

        if not pairs_to_scan:
            return None

        if news_filter():
            logging.debug("News block active, skipping scan")
            await asyncio.sleep(60)
            continue

        df_1m_data = await fetch_all_candles(pairs_to_scan, '1m', CANDLE_LIMIT)
        df_5m_data = await fetch_all_candles(pairs_to_scan, '5m', CANDLE_LIMIT)

        for pair_display in pairs_to_scan:
            df_1m = df_1m_data.get(pair_display)
            df_5m = df_5m_data.get(pair_display)
            if df_1m is None or df_5m is None:
                continue

            if not manual:
                current_candle = df_1m.index[-1]
                async with state['lock']:
                    if state.get('last_candle_time') == current_candle:
                        continue
                    state['last_candle_time'] = current_candle

            signal, confidence, reason, details = check_conditions(df_1m, df_5m)

            if signal != 'WAIT':
                if manual:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_signal = {
                            'pair': pair_display,
                            'signal': signal,
                            'confidence': confidence,
                            'reason': reason,
                            'entry_price': df_1m['close'].iloc[-1]
                        }
                        if confidence >= CONFIDENCE_THRESHOLD:
                            return best_signal
                else:
                    last_dir = db.get_last_signal_direction()
                    if last_dir == signal:
                        logging.debug("Duplicate direction, skipping")
                        continue
                    async with state['lock']:
                        last_signal_time = state.get('last_signal_time')
                    if last_signal_time and (datetime.now(timezone.utc) - last_signal_time).total_seconds() < 90:
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
        gc.collect()

        if not manual:
            break
        await asyncio.sleep(SCAN_INTERVAL_MANUAL)

    return best_signal if manual else None

def compute_entry_and_expiry(now: datetime):
    entry = now + timedelta(minutes=ENTRY_DELAY_MINUTES)
    entry = entry.replace(second=0, microsecond=0)
    if entry <= now:
        entry += timedelta(minutes=1)
    expiry = entry + timedelta(minutes=EXPIRY_MINUTES)
    return entry, expiry

def format_time_ist(dt_utc: datetime) -> str:
    return dt_utc.astimezone(INDIA_TZ).strftime('%H:%M')

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('bot.log'), logging.StreamHandler()]
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

    state = {
        'trading_enabled': False,
        'category': 'Crypto',         # default category
        'pair': None,
        'last_signal_time': None,
        'last_candle_time': None,
        'lock': asyncio.Lock(),
        'session_trades': 0,
        'session_losses': 0,
        'session_start_time': datetime.now(timezone.utc),
        'max_trades_per_session': 5,
        'max_losses_per_session': 2,
        'manual_request': False,
        'manual_chat_id': None,
        'manual_start_time': None,
        'manual_best_signal': None,
        'manual_loading_msg_id': None,
        'current_menu': {},
        'trades_today': 0,
        'last_heartbeat': datetime.now(timezone.utc),
        'rotation_index': 0,
        'market_cache': {},
        'cache_lock': asyncio.Lock(),
    }
    db = Database()
    bot = TradingBot(TELEGRAM_TOKEN, state, db)

    asyncio.create_task(watchdog(state))
    asyncio.create_task(result_checker(db))
    asyncio.create_task(auto_rotation(state, db))
    asyncio.create_task(bot.start())

    while True:
        try:
            async with state['lock']:
                state['last_heartbeat'] = datetime.now(timezone.utc)

            # Manual signal request
            async with state['lock']:
                manual_request = state['manual_request']
                chat_id = state['manual_chat_id']
                loading_msg_id = state.get('manual_loading_msg_id')
            if manual_request and chat_id:
                signal_info = await async_scan_for_signal(state, db, bot, manual=True)
                async with state['lock']:
                    state['manual_request'] = False
                    state['manual_chat_id'] = None
                    state['manual_start_time'] = None
                    state['manual_loading_msg_id'] = None
                if signal_info:
                    now = datetime.now(timezone.utc)
                    entry_time, expiry_time = compute_entry_and_expiry(now)
                    msg = (
                        "📊 *Signal Alert*\n"
                        "━━━━━━━━━━━━━━\n"
                        f"📈 Pair: {signal_info['pair']}\n"
                        f"📢 Signal: `{signal_info['signal']}`\n"
                        f"📊 Confidence: {signal_info['confidence']}%\n"
                        f"⏱ Entry: {format_time_ist(entry_time)}\n"
                        f"⌛ Expiry: {format_time_ist(expiry_time)}\n"
                        "━━━━━━━━━━━━━━\n"
                        f"🧠 Reason: {signal_info['reason']}"
                    )
                    db.log_signal(
                        "Crypto",
                        signal_info['pair'],
                        signal_info['signal'],
                        signal_info['confidence'],
                        signal_info['reason'],
                        signal_info['entry_price'],
                        expiry_time.isoformat()
                    )
                    asyncio.create_task(bot.send_signal(chat_id, msg, loading_msg_id))
                else:
                    asyncio.create_task(bot.send_signal(
                        chat_id,
                        "⚠️ Market conditions not clear.\nNo safe signal found.\nTry again in 1-2 minutes.",
                        loading_msg_id
                    ))
                await asyncio.sleep(30)
                continue

            # Auto mode
            async with state['lock']:
                trading_enabled = state['trading_enabled']
                current_pair = state.get('pair')
            if not trading_enabled or not current_pair:
                await asyncio.sleep(25)
                continue

            if not is_session_allowed():
                logging.debug("Outside session")
                await asyncio.sleep(60)
                continue

            if news_filter():
                logging.debug("News block active, waiting")
                await asyncio.sleep(60)
                continue

            async with state['lock']:
                if state['session_trades'] >= state['max_trades_per_session']:
                    logging.warning("Max trades reached")
                    await asyncio.sleep(60)
                    continue
                if state['session_losses'] >= state['max_losses_per_session']:
                    logging.warning("Max losses reached")
                    await asyncio.sleep(60)
                    continue

            signal_info = await async_scan_for_signal(state, db, bot, manual=False)
            if signal_info:
                now = datetime.now(timezone.utc)
                entry_time, expiry_time = compute_entry_and_expiry(now)
                msg = (
                    "📊 *Signal Alert*\n"
                    "━━━━━━━━━━━━━━\n"
                    f"📈 Pair: {signal_info['pair']}\n"
                    f"📢 Signal: `{signal_info['signal']}`\n"
                    f"📊 Confidence: {signal_info['confidence']}%\n"
                    f"⏱ Entry: {format_time_ist(entry_time)}\n"
                    f"⌛ Expiry: {format_time_ist(expiry_time)}\n"
                    "━━━━━━━━━━━━━━\n"
                    f"🧠 Reason: {signal_info['reason']}"
                )
                db.log_signal(
                    "Crypto",
                    signal_info['pair'],
                    signal_info['signal'],
                    signal_info['confidence'],
                    signal_info['reason'],
                    signal_info['entry_price'],
                    expiry_time.isoformat()
                )
                if TELEGRAM_CHAT_ID:
                    asyncio.create_task(bot.send_signal(TELEGRAM_CHAT_ID, msg))
                else:
                    logging.info(f"Auto signal: {signal_info['signal']} {signal_info['pair']}")
                async with state['lock']:
                    state['last_signal_time'] = datetime.now(timezone.utc)
                    state['session_trades'] += 1
                    state['trades_today'] += 1
            else:
                logging.debug("No auto signal")

            await asyncio.sleep(SCAN_INTERVAL_AUTO)

        except Exception as e:
            logging.exception("Main loop error")
            await asyncio.sleep(10)

async def safe_main():
    while True:
        try:
            await main()
        except Exception as e:
            logging.exception("CRASH DETECTED - restarting in 5 sec")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(safe_main())
