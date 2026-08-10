import os
import time
import threading
import requests
import pandas as pd
from flask import Flask

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

EMA_PERIOD = 288
RSI_PERIOD = 14
PROXIMITY_PRIMARY_PCT = 5.0    # Primary Entry zone (<= 5.0% or crossed)
PROXIMITY_WATCHLIST_MIN = 5.01 # Near-miss zone start
PROXIMITY_WATCHLIST_MAX = 7.00 # Near-miss zone end
EMA_30M_TOLERANCE_PCT = 5.0    # 30M EMA alignment tolerance (5.0%)

DIV_LOOKBACK_MIN = 5           # Expanded min lookback
DIV_LOOKBACK_MAX = 40          # Expanded max lookback
MIN_CANDLES_REQUIRED = 100     # Lowered to include newer tokens
HEARTBEAT_INTERVAL = 7200      # 2 Hours in seconds

COINDCX_PUBLIC_API = "https://public.coindcx.com"
COINDCX_TICKER_API = "https://api.coindcx.com/exchange/ticker"

# Flask Web Server for Render & Ping/Test Routes
app = Flask(__name__)

# Track global metrics for status & Heartbeat reports
last_scan_time = "Not started"
last_markets_count = 0
last_scan_duration = 0
last_primary_alerts_count = 0
watchlist_tokens = []
scanner_started = False

# ==========================================
# TELEGRAM NOTIFIER WITH CHAR LIMIT SPLITTING
# ==========================================
def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Error] Missing Telegram Bot Token or Chat ID.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    if len(message) <= 4000:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"[Error] Telegram alert failed: {e}")
    else:
        chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
        for idx, chunk in enumerate(chunks):
            chunk_msg = f"{chunk}\n\n*(Part {idx+1}/{len(chunks)})*"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk_msg, "parse_mode": "Markdown"}
            try:
                requests.post(url, json=payload, timeout=10)
                time.sleep(1)
            except Exception as e:
                print(f"[Error] Telegram chunk alert failed for part {idx+1}: {e}")

# ==========================================
# TECHNICAL ANALYSIS CALCULATIONS
# ==========================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_bearish_divergence(df: pd.DataFrame) -> bool:
    curr_idx = len(df) - 1
    curr_price = df['high'].iloc[curr_idx]
    curr_rsi = df['rsi'].iloc[curr_idx]

    lookback_df = df.iloc[max(0, curr_idx - DIV_LOOKBACK_MAX) : max(0, curr_idx - DIV_LOOKBACK_MIN)]
    if lookback_df.empty:
        return False

    past_max_idx = lookback_df['high'].idxmax()
    past_price = df['high'].loc[past_max_idx]
    past_rsi = df['rsi'].loc[past_max_idx]

    return (curr_price > past_price) and (curr_rsi < past_rsi)

def check_bullish_divergence(df: pd.DataFrame) -> bool:
    curr_idx = len(df) - 1
    curr_price = df['low'].iloc[curr_idx]
    curr_rsi = df['rsi'].iloc[curr_idx]

    lookback_df = df.iloc[max(0, curr_idx - DIV_LOOKBACK_MAX) : max(0, curr_idx - DIV_LOOKBACK_MIN)]
    if lookback_df.empty:
        return False

    past_min_idx = lookback_df['low'].idxmin()
    past_price = df['low'].loc[past_min_idx]
    past_rsi = df['rsi'].loc[past_min_idx]

    return (curr_price < past_price) and (curr_rsi > past_rsi)

# ==========================================
# COINDCX API DATA FETCHERS WITH RETRIES
# ==========================================
def get_all_pairs() -> list:
    try:
        res = requests.get(COINDCX_TICKER_API, timeout=10)
        if res.status_code == 200:
            return [item['market'] for item in res.json() if 'market' in item]
    except Exception as e:
        print(f"[Error] Fetching markets: {e}")
    return []

def fetch_candles(pair: str, interval: str, limit: int = 350, retries: int = 2) -> pd.DataFrame:
    url = f"{COINDCX_PUBLIC_API}/market_data/candles"
    for attempt in range(retries):
        try:
            res = requests.get(url, params={"pair": pair, "interval": interval, "limit": limit}, timeout=10)
            if res.status_code == 200 and len(res.json()) > 0:
                df = pd.DataFrame(res.json())
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
                return df.iloc[::-1].reset_index(drop=True)
        except Exception:
