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

# Proximity Bands
PROXIMITY_PRIMARY_PCT = 5.0    # Primary Entry zone (<= 5.0% or crossed)
PROXIMITY_WATCHLIST_MIN = 5.01 # Near-miss zone start
PROXIMITY_WATCHLIST_MAX = 10.00# Near-miss zone end
EMA_30M_TOLERANCE_PCT = 5.0    # 30M EMA alignment tolerance (5.0%)

# Lookback Window: From immediate candle (1 hour back) up to 40 hours back
DIV_LOOKBACK_MIN = 1           # Evaluates right from recent/current candle
DIV_LOOKBACK_MAX = 40          # Evaluates up to 40 hours back
MIN_RSI_DIFF = 0.5             # Ultra-sensitive RSI delta threshold
MIN_CANDLES_REQUIRED = 100     # Reduced candle count threshold to include newer pairs
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
    """Evaluates Bearish Divergence from live candle against past 1-40 candles."""
    curr_idx = len(df) - 1
    curr_price = df['high'].iloc[curr_idx]
    curr_rsi = df['rsi'].iloc[curr_idx]

    start_idx = max(0, curr_idx - DIV_LOOKBACK_MAX)
    end_idx = max(0, curr_idx - DIV_LOOKBACK_MIN)

    for i in range(start_idx, end_idx):
        past_price = df['high'].iloc[i]
        past_rsi = df['rsi'].iloc[i]

        if (curr_price > past_price) and ((past_rsi - curr_rsi) >= MIN_RSI_DIFF):
            return True

    return False

def check_bullish_divergence(df: pd.DataFrame) -> bool:
    """Evaluates Bullish Divergence from live candle against past 1-40 candles."""
    curr_idx = len(df) - 1
    curr_price = df['low'].iloc[curr_idx]
    curr_rsi = df['rsi'].iloc[curr_idx]

    start_idx = max(0, curr_idx - DIV_LOOKBACK_MAX)
    end_idx = max(0, curr_idx - DIV_LOOKBACK_MIN)

    for i in range(start_idx, end_idx):
        past_price = df['low'].iloc[i]
        past_rsi = df['rsi'].iloc[i]

        if (curr_price < past_price) and ((curr_rsi - past_rsi) >= MIN_RSI_DIFF):
            return True

    return False

# ==========================================
# COINDCX API DATA FETCHERS WITH SYMBOL FALLBACKS
# ==========================================
def get_all_pairs() -> list:
    """Fetches all active markets directly from CoinDCX live ticker API."""
    try:
        res = requests.get(COINDCX_TICKER_API, timeout=10)
        if res.status_code == 200:
            markets = [item['market'] for item in res.json() if 'market' in item]
            return list(set(markets))
    except Exception as e:
        print(f"[Error] Fetching markets list: {e}")
    return []

def fetch_candles(pair: str, interval: str, limit: int = 350) -> pd.DataFrame:
    """Fetches candle data with symbol format fallbacks (e.g., B-PAXG_USDT vs PAXGUSDT)."""
    url = f"{COINDCX_PUBLIC_API}/market_data/candles"
    
    # Generate list of possible symbol variants for robustness
    symbol_variants = [pair]
    if pair.startswith("B-"):
        symbol_variants.append(pair[2:])
    elif "_" not in pair and pair.endswith("USDT"):
        symbol_variants.append(f"B-{pair[:-4]}_USDT")
        symbol_variants.append(f"{pair[:-4]}_USDT")

    for symbol in symbol_variants:
        for attempt in range(2):
            try:
                res = requests.get(url, params={"pair": symbol, "interval": interval, "limit": limit}, timeout=8)
                if res.status_code == 200 and len(res.json()) > 0:
                    df = pd.DataFrame(res.json())
                    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
                    return df.iloc[::-1].reset_index(drop=True)
            except Exception:
                pass
            time.sleep(0.05)

    return pd.DataFrame()

# ==========================================
# SCANNER CORE LOGIC
# ==========================================
def process_market(pair: str, is_diagnostic: bool = False) -> dict:
    global watchlist_tokens

    df_1h = fetch_candles(pair, interval="1h")
    df_30m = fetch_candles(pair, interval="30m")

    if df_1h.empty or len(df_1h) < MIN_CANDLES_REQUIRED or df_30m.empty or len(df_30m) < MIN_CANDLES_REQUIRED:
        return {"status": "skipped"}

    df_1h['ema288'] = df_1h['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_1h['rsi'] = calculate_rsi(df_1h['close'], period=RSI_PERIOD)
    df_30m['ema288'] = df_30m['close'].ewm(span=EMA_PERIOD, adjust=False).mean()

    live_price = df_1h.iloc[-1]['close']
    ema_1h = df_1h.iloc[-1]['ema288']
    ema_30m = df_30m.iloc[-1]['ema288']

    dist_pct = abs(live_price - ema_1h) / ema_1h * 100
    ema_30m_diff_pct = (ema_30m - ema_1h) / ema_1h * 100

    # ------------------------------------
    # BEARISH SETUP EVALUATION
    # ------------------------------------
    if check_bearish_divergence(df_1h):
        is_30m_aligned = ema_30m >= (ema_1h * (1 - EMA_30M_TOLERANCE_PCT / 100))
        
        if dist_pct <= PROXIMITY_PRIMARY_PCT or live_price >= ema_1h:
            tier_badge = "⭐ *A+ BEARISH CONFLUENCE SETUP*" if is_30m_aligned else "⚡ *1H BEARISH PRIMARY SETUP (30M Extended)*"
            status_note = f"✅ 30M EMA within 5.0% tolerance ({round(ema_30m_diff_pct, 2)}%)" if is_30m_aligned else f"⚠️ 30M EMA is {abs(round(ema_30m_diff_pct, 2))}% below 1H EMA"

            if not is_diagnostic:
                msg = (
                    f"{tier_badge}\n"
                    f"------------------------------------\n"
                    f"• *Market:* {pair}\n"
                    f"• *Live Price:* {live_price}\n"
                    f"• *1H 288 EMA:* {round(ema_1h, 4)} (Dist: {round(dist_pct, 2)}%)\n"
                    f"• *1H Signal:* Bearish RSI Divergence (1–40 candles)\n\n"
                    f"📊 *TIMEFRAME CONFLUENCE:*\n"
                    f"• *30M 288 EMA:* {round(ema_30m, 4)}\n"
                    f"• *30M Status:* {status_note}"
                )
                send_telegram_alert(msg)

            return {
                "status": "primary_alert",
                "pair": pair,
                "type": "🔴 Bearish",
                "tier": "A+" if is_30m_aligned else "1H",
                "price": live_price,
                "ema_1h": round(ema_1h, 4),
                "dist": round(dist_pct, 2)
            }

        elif PROXIMITY_WATCHLIST_MIN <= dist_pct <= PROXIMITY_WATCHLIST_MAX:
            token_info = {
                "pair": pair,
                "type": "🔴 Bearish",
                "price": live_price,
                "ema_1h": round(ema_1h, 4),
                "dist": round(dist_pct, 2)
            }
            if not is_diagnostic:
                watchlist_tokens.append(token_info)
            return {"status": "near_miss", "info": token_info}

    # ------------------------------------
    # BULLISH SETUP EVALUATION
    # ------------------------------------
    if check_bullish_divergence(df_1h):
        is_30m_aligned = ema_30m <= (ema_1h * (1 + EMA_30M_TOLERANCE_PCT / 100))

        if dist_pct <= PROXIMITY_PRIMARY_PCT or live_price <= ema_1h:
            tier_badge = "⭐ *A+ BULLISH CONFLUENCE SETUP*" if is_30m_aligned else "⚡ *1H BULLISH PRIMARY SETUP (30M Extended)*"
            status_note = f"✅ 30M EMA within 5.0% tolerance ({round(ema_30m_diff_pct, 2)}%)" if is_30m_aligned else f"⚠️ 30M EMA is {abs(round(ema_30m_diff_pct, 2))}% above 1H EMA"

            if not is_diagnostic:
                msg = (
                    f"{tier_badge}\n"
                    f"------------------------------------\n"
                    f"• *Market:* {pair}\n"
                    f"• *Live Price:* {live_price}\n"
                    f"• *1H 288 EMA:* {round(ema_1h, 4)} (Dist: {round(dist_pct, 2)}%)\n"
                    f"• *1H Signal:* Bullish RSI Divergence (1–40 candles)\n\n"
                    f"📊 *TIMEFRAME CONFLUENCE:*\n"
                    f"• *30M 288 EMA:* {round(ema_30m, 4)}\n"
                    f"• *30M Status:* {status_note}"
                )
                send_telegram_alert(msg)

            return {
                "status": "primary_alert",
                "pair": pair,
                "type": "🟢 Bullish",
                "tier": "A+" if is_30m_aligned else "1H",
                "price": live_price,
                "ema_1h": round(ema_1h, 4),
                "dist": round(dist_pct, 2)
            }

        elif PROXIMITY_WATCHLIST_MIN <= dist_pct <= PROXIMITY_WATCHLIST_MAX:
            token_info = {
                "pair": pair,
                "type": "🟢 Bullish",
                "price": live_price,
                "ema_1h": round(ema_1h, 4),
                "dist": round(dist_pct, 2)
            }
            if not is_diagnostic:
                watchlist_tokens.append(token_info)
            return {"status": "near_miss", "info": token_info}

    return {"status": "normal"}

# ==========================================
# DIAGNOSTIC MANUAL TEST RUNNER
# ==========================================
def run_diagnostic_test():
    start_time = time.time()
    markets = get_all_pairs()
    total_scanned = len(markets)
    primary_alerts_found = 0
    near_miss_list = []

    for pair in markets:
        res = process_market(pair, is_diagnostic=True)
        if res["status"] == "primary_alert":
            primary_alerts_found += 1
        elif res["status"] == "near_miss":
            near_miss_list.append(res["info"])
        time.sleep(0.04)

    duration = time.time() - start_time
    time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    msg_lines = [
        "🧪 *MANUAL DIAGNOSTIC SCAN REPORT*",
        "------------------------------------",
        f"• *System Health:* 🟢 Alive & Operational",
        f"• *Total Markets Scanned:* {total_scanned}",
        f"• *Primary Alerts Triggered (<= 5.0%):* {primary_alerts_found}",
        f"• *Near-Miss Tokens Found (5.01% - 10.0%):* {len(near_miss_list)}",
        f"• *Scan Duration:* {round(duration, 1)}s",
        f"• *Timestamp:* {time_str}"
    ]

    if near_miss_list:
        msg_lines.append("\n👀 *NEAR-MISS WATCHLIST (5.01% - 10.00%)*")
        msg_lines.append("------------------------------------")
        for t in near_miss_list:
            msg_lines.append(f"• {t['type']} *{t['pair']}* | Dist: {t['dist']}% | Price: {t['price']}")
    else:
        msg_lines.append("\n• _No tokens currently sitting in the 5.01%-10.00% near-miss zone._")

    send_telegram_alert("\n".join(msg_lines))

# ==========================================
# 2-HOUR HEARTBEAT & WATCHLIST
# ==========================================
def send_2hour_heartbeat():
    global watchlist_tokens, last_primary_alerts_count
    
    msg_lines = [
        "💓 *2-HOUR SYSTEM HEARTBEAT & WATCHLIST*",
        "------------------------------------",
        f"• *Status:* 🟢 Bot Active & Scanning",
        f"• *Total CoinDCX Pairs Scanned:* {last_markets_count}",
        f"• *Primary Alerts Triggered (Last Scan):* {last_primary_alerts_count}",
        f"• *Scan Duration:* {round(last_scan_duration, 1)}s",
        f"• *Last Completed Scan:* {last_scan_time}",
        "\n👀 *NEAR-MISS WATCHLIST (Proximity 5.01% - 10.00%)*",
        "------------------------------------"
    ]

    if not watchlist_tokens:
        msg_lines.append("• _No near-miss tokens currently in the 5.01%-10.00% zone with RSI divergence._")
    else:
        for t in watchlist_tokens:
            msg_lines.append(
                f"• {t['type']} *{t['pair']}*\n"
                f"  - Distance: {t['dist']}% | Price: {t['price']} | 1H EMA: {t['ema_1h']}"
            )

    msg_lines.append("\n⚠️ *Keep an eye on these tokens as they approach your 5.00% entry zone!*")
    send_telegram_alert("\n".join(msg_lines))

# ==========================================
# BACKGROUND CONTINUOUS SCANNER THREAD
# ==========================================
def scanner_loop():
    global last_scan_time, last_markets_count, last_scan_duration, watchlist_tokens, last_primary_alerts_count

    startup_msg = (
        "🚀 *CoinDCX Scanner Bot Started & Active!*\n\n"
        "• *Status:* 🟢 Online in Render Cloud\n"
        "• *Primary Entry Zone:* <= 5.0%\n"
        "• *Near-Miss Watchlist:* 5.01% - 10.00%\n"
        "• *Divergence Window:* 1–40 Candles (Recent Candle Inclusive)\n"
        "• *30M EMA Tolerance:* 5.0%"
    )
    send_telegram_alert(startup_msg)

    last_heartbeat_time = time.time()

    while True:
        try:
            start_time = time.time()
            watchlist_tokens = []
            current_cycle_primary_count = 0

            markets = get_all_pairs()
            last_markets_count = len(markets)

            for pair in markets:
                res = process_market(pair)
                if res and res.get("status") == "primary_alert":
                    current_cycle_primary_count += 1
                time.sleep(0.04)

            last_primary_alerts_count = current_cycle_primary_count
            last_scan_duration = time.time() - start_time
            last_scan_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            print(f"[Scan Complete] Scanned {last_markets_count} pairs in {round(last_scan_duration, 1)}s | Primary Alerts: {last_primary_alerts_count}")

            if time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                send_2hour_heartbeat()
                last_heartbeat_time = time.time()

            time.sleep(90)

        except Exception as e:
            print(f"[Error in main scanner loop]: {e}")
            time.sleep(60)

def start_scanner_once():
    global scanner_started
    if not scanner_started:
        scanner_started = True
        t = threading.Thread(target=scanner_loop, daemon=True)
        t.start()

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def home():
    start_scanner_once()
    return f"🟢 CoinDCX Scanner Web Service Active! Last scan: {last_scan_time}", 200

@app.route('/test')
def trigger_test():
    start_scanner_once()
    threading.Thread(target=run_diagnostic_test).start()
    return "<h1>🟢 Manual Diagnostic Test Triggered! Check your Telegram app for the full report.</h1>", 200

# Start background thread automatically
start_scanner_once()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
