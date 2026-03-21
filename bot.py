#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Binary Options Trading Signal Bot - Final Version
- Entry time buffer (2 min normal, 1 min turbo)
- Times displayed in Indian Standard Time (IST)
- Full Telegram button control, back navigation everywhere
- 50+ pairs, auto rotation, strongest pair selection
- News avoidance, volatility filters, turbo mode
- Auto‑restart, watchdog, result tracking
- Optimized for Termux (low CPU, low memory)
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
DATA_TIMEOUT = 6
BINANCE_BASE_URL = "https://api.binance.com/api/v3"

# Timezone for display (Indian Standard Time)
INDIA_TZ = ZoneInfo("Asia/Kolkata")

# ======================= CATEGORIES & PAIRS =======================
CATEGORIES = {
    "Currencies": [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
        "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "CHF/JPY", "EUR/AUD", "GBP/AUD",
        "EUR/CAD", "GBP/CAD", "AUD/CAD", "NZD/JPY", "USD/SGD", "USD/HKD"
    ],
    "Crypto": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT",
        "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT", "TRXUSDT", "ETCUSDT", "XLMUSDT",
        "NEARUSDT", "ATOMUSDT", "LINKUSDT", "UNIUSDT", "FILUSDT", "ICPUSDT"
    ],
    "Commodities": [
        "Gold", "Silver", "WTI Oil", "Brent Oil", "Natural Gas", "Copper"
    ],
    "Stocks": [
        "AAPL", "TSLA", "AMZN", "MSFT", "GOOGL", "META", "NVDA", "NFLX", "AMD", "INTC",
        "PYPL", "BABA", "UBER", "DIS", "KO"
    ]
}

# ======================= UNIVERSAL SYMBOL NORMALIZATION =======================
def normalize_symbol(pair_name: str) -> str:
    pair_name = pair_name.upper().strip()
    pair_name = pair_name.replace(" ", "")
    pair_name = pair_name.replace("(OTC)", "")
    if "/" in pair_name:
        base, quote = pair_name.split("/")
        return f"{base}{quote}"
    if pair_name == "GOLD":
        return "XAUUSDT"
    if pair_name == "SILVER":
        return "XAGUSDT"
    if pair_name == "WTIOIL":
        return "WTI"
    if pair_name == "BRENTOIL":
        return "Brent"
    if pair_name == "NATURALGAS":
        return "NG"
    if pair_name == "COPPER":
        return "COPPER"
    pair_name = pair_name.replace("-", "")
    return pair_name

DISPLAY_TO_SYMBOL = {}
for category, pairs in CATEGORIES.items():
    for display in pairs:
        DISPLAY_TO_SYMBOL[display] = normalize_symbol(display)

# ======================= TIME HELPERS =======================
def format_time_ist(dt_utc: datetime) -> str:
    local = dt_utc.astimezone(INDIA_TZ)
    return local.strftime('%H:%M')

# ======================= PERFORMANCE SETTINGS =======================
MAX_CONCURRENT_FETCH = 2
CANDLE_LIMIT = 40
SCAN_INTERVAL_AUTO = 8
SCAN_INTERVAL_MANUAL = 2.5
CONFIDENCE_THRESHOLD_AUTO = 75
CONFIDENCE_THRESHOLD_MANUAL = 70
FALLBACK_CONFIDENCE = 50

TURBO_CONFIDENCE_THRESHOLD = 60
TURBO_SCAN_INTERVAL = 1.5
TURBO_EXPIRY_MINUTES = 1          # expiry duration after entry
ENTRY_DELAY_MINUTES = 2            # delay before entry (normal mode)
TURBO_ENTRY_DELAY_MINUTES = 1      # delay before entry (turbo mode)

MAX_SCAN_PAIRS_MANUAL = 6

NEWS_BLOCKS = [
    (13, 25, 13, 40),
    (15, 55, 16, 10)
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
            symbol = DISPLAY_TO_SYMBOL.get(pair)
            if symbol is None:
                logging.error(f"No symbol mapping for {pair}")
                continue
            tasks.append(fetch_candles_async(session, symbol, interval, limit))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for pair, df in zip(pairs, results):
        if isinstance(df, pd.DataFrame) and not df.empty:
            data[pair] = df
    return data

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

def check_conditions(df_1m: pd.DataFrame, df_5m: pd.DataFrame, mode: str = 'auto', turbo: bool = False) -> Tuple[str, int, str, Dict]:
    details = {}
    if df_1m.empty or df_5m.empty:
        return 'WAIT', 0, "Missing data", details
    if len(df_1m) < 30 or len(df_5m) < 30:
        return 'WAIT', 0, "Insufficient data", details

    close_1m = df_1m['close']
    ema9_1m = compute_ema(close_1m, 9)
    ema21_1m = compute_ema(close_1m, 21)
    rsi_1m = compute_rsi(close_1m, 14)
    upper_1m, middle_1m, lower_1m = compute_bollinger_bands(close_1m, 20, 2)
    macd_line_1m, signal_line_1m, hist_1m = compute_macd(close_1m, 12, 26, 9)
    atr_1m = compute_atr(df_1m, 14)
    candle_type, candle_score = candlestick_strength(df_1m)

    close_5m = df_5m['close']
    ema9_5m = compute_ema(close_5m, 9)
    ema21_5m = compute_ema(close_5m, 21)

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

    if 45 <= rsi_1m <= 55:
        return 'WAIT', 0, "RSI neutral (45-55)", details

    if details['bb_width'] < 0.003:
        return 'WAIT', 0, "Low volatility (BB too tight)", details
    if details['atr_ratio'] < 0.0015:
        return 'WAIT', 0, "Low movement (ATR too small)", details

    if mode == 'auto':
        if len(df_1m) >= 20:
            avg_vol = df_1m['volume'].tail(20).mean()
            if df_1m['volume'].iloc[-1] < avg_vol * 0.7:
                return 'WAIT', 0, "Low volume", details

    if mode == 'auto' and atr_1m is not None:
        price_change = abs(close_1m.iloc[-1] - close_1m.iloc[-2]) / close_1m.iloc[-2] * 100
        required_move = details['bb_width'] * 0.1 if details['bb_width'] else 0.05
        if price_change < required_move:
            return 'WAIT', 0, f"Movement too small ({price_change:.2f}%)", details

    confidence = 0
    reasons = []

    if (trend == 'bullish' and ema9_1m > ema21_1m) or (trend == 'bearish' and ema9_1m < ema21_1m):
        confidence += 25
        reasons.append("EMA aligned")
    else:
        return 'WAIT', 0, "EMA condition failed", details

    if trend == 'bullish' and 40 <= rsi_1m <= 65:
        confidence += 20
        reasons.append("RSI bullish range")
    elif trend == 'bearish' and 35 <= rsi_1m <= 60:
        confidence += 20
        reasons.append("RSI bearish range")
    else:
        return 'WAIT', 0, f"RSI out of range: {rsi_1m:.2f}", details

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
    else:
        if trend == 'bullish' and price <= middle_1m * 1.02:
            confidence += 15
            reasons.append("Price near lower/mid BB")
        elif trend == 'bearish' and price >= middle_1m * 0.98:
            confidence += 15
            reasons.append("Price near upper/mid BB")
        else:
            return 'WAIT', 0, "BB position mismatch (manual)", details

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
        if mode == 'manual' and ((trend == 'bullish' and hist_1m > 0 and macd_line_1m > signal_line_1m) or
                                 (trend == 'bearish' and hist_1m < 0 and macd_line_1m < signal_line_1m)):
            confidence += 10
            reasons.append("MACD already in momentum")
        else:
            return 'WAIT', 0, "MACD condition missing", details

    if mode == 'auto':
        if (trend == 'bullish' and candle_type == 'bullish' and candle_score >= 70) or \
           (trend == 'bearish' and candle_type == 'bearish' and candle_score >= 70):
            confidence += 15
            reasons.append(f"Strong {candle_type} candle")
        else:
            return 'WAIT', 0, f"Candle weak: {candle_type} ({candle_score})", details
    else:
        if (trend == 'bullish' and candle_type == 'bullish' and candle_score >= 60) or \
           (trend == 'bearish' and candle_type == 'bearish' and candle_score >= 60):
            confidence += 10
            reasons.append(f"Moderate {candle_type} candle")
        else:
            return 'WAIT', 0, f"Candle weak: {candle_type} ({candle_score})", details

    if atr_1m is not None and (atr_1m / price) > 0.008:
        confidence += 5
        reasons.append("Strong volatility")
    if details['atr_ratio'] > 0.01:
        confidence += 8
        reasons.append("High volatility boost")
    if details['trend_strength'] > 0.25:
        confidence += 5
        reasons.append("Strong trend strength")
    if details['bb_width'] > 0.012:
        confidence += 5
        reasons.append("Wide Bollinger Bands")

    signal = 'CALL' if trend == 'bullish' else 'PUT'
    threshold = CONFIDENCE_THRESHOLD_AUTO if mode == 'auto' else (TURBO_CONFIDENCE_THRESHOLD if turbo else CONFIDENCE_THRESHOLD_MANUAL)
    if confidence >= threshold:
        return signal, min(confidence, 100), ", ".join(reasons), details
    else:
        return 'WAIT', confidence, f"Confidence {confidence} < {threshold}", details

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
            [InlineKeyboardButton("⚡ Turbo 1m", callback_data="turbo_mode")],
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
        pairs = CATEGORIES[category]
        keyboard = []
        for i in range(0, len(pairs), 2):
            row = []
            for j in range(i, min(i+2, len(pairs))):
                row.append(InlineKeyboardButton(pairs[j], callback_data=f"pair_{pairs[j]}"))
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
            turbo = self.state.get('turbo_mode', False)
        mode = "Auto" if trading else "Manual (inactive)"
        turbo_text = " (Turbo)" if turbo else ""
        text = (
            "📊 *Bot Status*\n"
            "━━━━━━━━━━━━━━\n"
            f"Mode: {mode}{turbo_text}\n"
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
                    "⚠️ Please select a category and pair first using Settings → Select Category → Select Pair.",
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

    async def update_timer(self, chat_id: int, msg_id: int, end_time: datetime, turbo: bool = False):
        interval = TURBO_SCAN_INTERVAL if turbo else 15
        while True:
            remaining = end_time - datetime.now(timezone.utc)
            seconds = int(remaining.total_seconds())
            if seconds <= 0:
                break
            minutes = seconds // 60
            secs = seconds % 60
            text = f"⏳ Scanning market...\n\nNext signal will start after delay\n\nTime remaining: {minutes:02}:{secs:02}\n\nYou will have time to execute trade"
            try:
                await self.app.bot.edit_message_text(
                    text,
                    chat_id=chat_id,
                    message_id=msg_id
                )
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def handle_generate_signal(self, chat_id: int, message_id: int):
        async with self.state['lock']:
            if self.state.get('pair') is None:
                await self.app.bot.edit_message_text(
                    "⚠️ Please select a category and pair first using Settings → Select Category → Select Pair.",
                    chat_id=chat_id, message_id=message_id
                )
                return
            turbo = self.state.get('turbo_mode', False)

        keyboard = [
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_generate")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_main")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        loading_msg = await self.app.bot.edit_message_text(
            "⏳ Scanning market...\n\nNext signal will start after delay\n\nYou will have time to execute trade",
            chat_id=chat_id, message_id=message_id, reply_markup=reply_markup
        )

        # Calculate end time for scanning (max 5 minutes from now)
        end_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        asyncio.create_task(self.update_timer(chat_id, loading_msg.message_id, end_time, turbo))

        async with self.state['lock']:
            self.state['manual_request'] = True
            self.state['manual_chat_id'] = chat_id
            self.state['manual_start_time'] = datetime.now(timezone.utc)
            self.state['manual_best_signal'] = None
            self.state['manual_loading_msg_id'] = loading_msg.message_id
            self.state['manual_end_time'] = end_time
            self.state['turbo_mode'] = turbo

    async def handle_cancel_generate(self, chat_id: int, message_id: int):
        async with self.state['lock']:
            self.state['manual_request'] = False
            self.state['manual_chat_id'] = None
            self.state['manual_start_time'] = None
            self.state['manual_best_signal'] = None
            self.state['manual_loading_msg_id'] = None
            self.state['manual_end_time'] = None
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
            self.state['pair'] = pair_display
        text = f"✅ Pair set to {pair_display} (Category: {category})."
        keyboard = [
            [InlineKeyboardButton("🔄 Change Pair", callback_data="select_pair")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id,
                                             reply_markup=reply_markup)

    async def handle_turbo_mode(self, chat_id: int, message_id: int):
        async with self.state['lock']:
            self.state['turbo_mode'] = not self.state.get('turbo_mode', False)
            turbo = self.state['turbo_mode']
        if turbo:
            text = "⚡ Turbo Mode ENABLED\n\n- Signals generated faster\n- 1-minute expiry after entry delay\n- Slightly relaxed confidence threshold\n- Entry delay reduced to 1 minute"
        else:
            text = "⚡ Turbo Mode DISABLED\n\nNormal mode restored (5-minute expiry, 2-minute entry delay, strict confidence)."
        keyboard = [
            [InlineKeyboardButton("⬅️ Back", callback_data="back_settings")],
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
        elif data == "turbo_mode":
            await self.handle_turbo_mode(chat_id, message_id)
        elif data.startswith("category_"):
            category = data[9:]
            await self.handle_category_selection(chat_id, message_id, category)
        elif data.startswith("pair_"):
            pair = data[5:]
            await self.handle_pair_selection(chat_id, message_id, pair)

    async def send_signal(self, chat_id: int, signal_text: str, loading_msg_id: int = None):
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

# ======================= WATCHDOG =======================
async def watchdog(state):
    while True:
        await asyncio.sleep(60)
        async with state['lock']:
            last = state.get('last_heartbeat')
        if last:
            delay = (datetime.now(timezone.utc) - last).total_seconds()
            if delay > 180:
                logging.warning("WATCHDOG ALERT: BOT FROZEN – no heartbeat for 3 minutes")

# ======================= RESULT CHECKER =======================
async def result_checker(db):
    while True:
        await asyncio.sleep(30)
        conn = sqlite3.connect("signals.db")
        c = conn.cursor()
        c.execute("""
            SELECT id, pair, signal, entry_price, expiry_time
            FROM signals
            WHERE result IS NULL
        """)
        rows = c.fetchall()
        for row in rows:
            id_, pair_display, signal, entry_price, expiry_str = row
            expiry = datetime.fromisoformat(expiry_str)
            if datetime.now(timezone.utc) < expiry:
                continue
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
        await asyncio.sleep(300)
        async with state['lock']:
            if not state['trading_enabled']:
                continue
            category = state.get('category')
            if not category:
                continue
            pairs = CATEGORIES[category]
            if not pairs:
                continue
            idx = state.get('rotation_index', 0)
            new_pair = pairs[idx % len(pairs)]
            state['rotation_index'] = idx + 1
            state['pair'] = new_pair
            logging.info(f"Auto rotation: switched to {new_pair}")

# ======================= MAIN LOOP =======================
async def async_scan_for_signal(state, db, bot, manual=False):
    start_time = datetime.now(timezone.utc)
    best_signal = None
    best_score = 0

    while True:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        if manual and elapsed > 300:
            if best_signal:
                return best_signal
            else:
                async with state['lock']:
                    pair = state.get('pair', 'BTC/USDT')
                return {
                    'pair': pair,
                    'signal': 'CALL',
                    'confidence': FALLBACK_CONFIDENCE,
                    'reason': 'Fallback signal (safe mode)',
                    'entry_price': 0
                }

        pairs_to_scan = []
        if not manual:
            async with state['lock']:
                pair = state.get('pair')
            if pair:
                pairs_to_scan = [pair]
            else:
                return None
        else:
            async with state['lock']:
                category = state.get('category')
            if category:
                all_pairs = CATEGORIES[category]
                pairs_to_scan = all_pairs[:MAX_SCAN_PAIRS_MANUAL]
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

            mode = 'manual' if manual else 'auto'
            turbo = state.get('turbo_mode', False) if manual else False
            signal, confidence, reason, details = check_conditions(df_1m, df_5m, mode, turbo)

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
                        threshold = TURBO_CONFIDENCE_THRESHOLD if turbo else CONFIDENCE_THRESHOLD_MANUAL
                        if confidence >= threshold:
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
        await asyncio.sleep(SCAN_INTERVAL_MANUAL if not state.get('turbo_mode') else TURBO_SCAN_INTERVAL)

    return best_signal if manual else None

# ======================= SIGNAL HELPER (Entry delay) =======================
def compute_entry_and_expiry(turbo: bool, now: datetime):
    delay = TURBO_ENTRY_DELAY_MINUTES if turbo else ENTRY_DELAY_MINUTES
    entry = now + timedelta(minutes=delay)
    # round to next minute
    entry = entry.replace(second=0, microsecond=0)
    if entry <= now:
        entry += timedelta(minutes=1)
    expiry_minutes = TURBO_EXPIRY_MINUTES if turbo else 5
    expiry = entry + timedelta(minutes=expiry_minutes)
    return entry, expiry

# ======================= MAIN =======================
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
        'category': None,
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
        'manual_end_time': None,
        'current_menu': {},
        'trades_today': 0,
        'last_heartbeat': datetime.now(timezone.utc),
        'rotation_index': 0,
        'turbo_mode': False,
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
                    state['manual_end_time'] = None
                if signal_info:
                    now = datetime.now(timezone.utc)
                    turbo = state.get('turbo_mode', False)
                    entry_time, expiry_time = compute_entry_and_expiry(turbo, now)
                    note = ""
                    threshold = TURBO_CONFIDENCE_THRESHOLD if turbo else CONFIDENCE_THRESHOLD_MANUAL
                    if signal_info['confidence'] < threshold:
                        note = "\n⚠️ *Best available signal (confidence below threshold)*"
                    msg = (
                        "📊 *Signal Alert*\n"
                        "━━━━━━━━━━━━━━\n"
                        f"📈 Pair: {signal_info['pair']}\n"
                        f"📢 Signal: `{signal_info['signal']}`\n"
                        f"📊 Confidence: {signal_info['confidence']}%\n"
                        f"⏱ Entry: {format_time_ist(entry_time)}\n"
                        f"⌛ Expiry: {format_time_ist(expiry_time)}\n"
                        "━━━━━━━━━━━━━━\n"
                        f"🧠 Reason: {signal_info['reason']}{note}"
                    )
                    category = None
                    for cat, pairs in CATEGORIES.items():
                        if signal_info['pair'] in pairs:
                            category = cat
                            break
                    db.log_signal(
                        category or "Unknown",
                        signal_info['pair'],
                        signal_info['signal'],
                        signal_info['confidence'],
                        signal_info['reason'],
                        signal_info['entry_price'],
                        expiry_time.isoformat()
                    )
                    asyncio.create_task(bot.send_signal(chat_id, msg, loading_msg_id))
                else:
                    asyncio.create_task(bot.send_signal(chat_id, "⚠️ Unexpected error: no signal found. Please try again.", loading_msg_id))
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
                entry_time, expiry_time = compute_entry_and_expiry(False, now)  # auto mode never uses turbo
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
                category = None
                for cat, pairs in CATEGORIES.items():
                    if signal_info['pair'] in pairs:
                        category = cat
                        break
                db.log_signal(
                    category or "Unknown",
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
