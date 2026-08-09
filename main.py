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
PROXIMITY_PRIMARY_PCT = 0.5    # Entry zone (<= 0.5% or crossed)
PROXIMITY_WATCHLIST_MIN = 0.51 # Near-miss zone start
PROXIMITY_WATCHLIST_MAX = 1.50 # Near-miss zone end
DIV_LOOKBACK_MIN = 15
DIV_LOOKBACK_MAX = 30
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

# ==========================================
# TELEGRAM NOTIFIER WITH CHAR LIMIT SPLITTING
# ==========================================
def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Error] Missing Telegram Bot Token or Chat ID.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Telegram message character limit is 4096 characters
    if len(message) <= 4000:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"[Error] Telegram alert failed: {e}")
    else:
        # Split message into chunks if it exceeds Telegram limits
        chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
        for idx, chunk in enumerate(chunks):
            chunk_msg = f"{chunk}\n\n*(Part {idx+1}/{len(chunks)})*"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk_msg, "parse_mode": "Markdown"}
            try:
                requests.post(url, json=payload, timeout=10)
                time.sleep(1)
            except Exception as e:
                print(f"[Error] Telegram chunk alert failed: {e}")

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

    lookback_df = df.iloc[curr_idx - DIV_LOOKBACK_MAX : curr_idx - DIV_LOOKBACK_MIN]
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

    lookback_df = df.iloc[curr_idx - DIV_LOOKBACK_MAX : curr_idx - DIV_LOOKBACK_MIN]
    if lookback_df.empty:
        return False

    past_min_idx = lookback_df['low'].idxmin()
    past_price = df['low'].loc[past_min_idx]
    past_rsi = df['rsi'].loc[past_min_idx]

    return (curr_price < past_price) and (curr_rsi > past_rsi)

# ==========================================
# COINDCX API DATA FETCHERS
# ==========================================
def get_all_pairs() -> list:
    try:
        res = requests.get(COINDCX_TICKER_API, timeout=10)
        if res.status_code == 200:
            return [item['market'] for item in res.json() if 'market' in item]
    except Exception as e:
        print(f"[Error] Fetching markets: {e}")
    return []

def fetch_candles(pair: str, interval: str, limit: int = 350) -> pd.DataFrame:
    url = f"{COINDCX_PUBLIC_API}/market_data/candles"
    try:
        res = requests.get(url, params={"pair": pair, "interval": interval, "limit": limit}, timeout=10)
        if res.status_code == 200 and len(res.json()) > 0:
            df = pd.DataFrame(res.json())
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
            return df.iloc[::-1].reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()

# ==========================================
# SCANNER CORE LOGIC
# ==========================================
def process_market(pair: str, is_diagnostic: bool = False) -> dict:
    global watchlist_tokens

    df_1h = fetch_candles(pair, interval="1h")
    df_30m = fetch_candles(pair, interval="30m")

    if df_1h.empty or len(df_1h) < 290 or df_30m.empty or len(df_30m) < 290:
        return {"status": "skipped"}

    df_1h['ema288'] = df_1h['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_1h['rsi'] = calculate_rsi(df_1h['close'], period=RSI_PERIOD)
    df_30m['ema288'] = df_30m['close'].ewm(span=EMA_PERIOD, adjust=False).mean()

    live_price = df_1h.iloc[-1]['close']
    ema_1h = df_1h.iloc[-1]['ema288']
    ema_30m = df_30m.iloc[-1]['ema288']

    dist_pct = abs(live_price - ema_1h) / ema_1h * 100

    # Bearish Setup
    if ema_30m >= (ema_1h * 0.9992):
        if check_bearish_divergence(df_1h):
            if dist_pct <= PROXIMITY_PRIMARY_PCT or live_price >= ema_1h:
                if not is_diagnostic:
                    msg = (
                        f"🔴 *PRIMARY BEARISH ALERT: {pair}*\n\n"
                        f"• *Live Price:* {live_price}\n"
                        f"• *1H 288 EMA:* {round(ema_1h, 4)}\n"
                        f"• *30M 288 EMA:* {round(ema_30m, 4)}\n"
                        f"• *Distance to EMA:* {round(dist_pct, 2)}%\n"
                        f"• *Signal:* Bearish RSI Divergence (15–30 candles) active near 1H EMA!"
                    )
                    send_telegram_alert(msg)
                return {"status": "primary_alert", "pair": pair, "type": "🔴 Bearish", "price": live_price, "ema_1h": round(ema_1h, 4), "dist": round(dist_pct, 2)}

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

    # Bullish Setup
    if ema_30m <= (ema_1h * 1.0008):
        if check_bullish_divergence(df_1h):
            if dist_pct <= PROXIMITY_PRIMARY_PCT or live_price <= ema_1h:
                if not is_diagnostic:
                    msg = (
                        f"🟢 *PRIMARY BULLISH ALERT: {pair}*\n\n"
                        f"• *Live Price:* {live_price}\n"
                        f"• *1H 288 EMA:* {round(ema_1h, 4)}\n"
                        f"• *30M 288 EMA:* {round(ema_30m, 4)}\n"
                        f"• *Distance to EMA:* {round(dist_pct, 2)}%\n"
                        f"• *Signal:* Bullish RSI Divergence (15–30 candles) active near 1H EMA!"
                    )
                    send_telegram_alert(msg)
                return {"status": "primary_alert", "pair": pair, "type": "🟢 Bullish", "price": live_price, "ema_1h": round(ema_1h, 4), "dist": round(dist_pct, 2)}

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
# DIAGNOSTIC MANUAL TEST RUNNER (VIA /test URL)
# ==========================================
def run_diagnostic_test():
    """Runs a manual scan pass and delivers detailed summary metrics to Telegram."""
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
        time.sleep(0.05)

    duration = time.time() - start_time
    time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    msg_lines = [
        "🧪 *MANUAL DIAGNOSTIC SCAN REPORT*",
        "------------------------------------",
        f"• *System Health:* 🟢 Alive & Operational",
        f"• *Total Markets Scanned:* {total_scanned}",
        f"• *Primary Alerts Triggered:* {primary_alerts_found}",
        f"• *Near-Miss Tokens Found:* {len(near_miss_list)}",
        f"• *Scan Duration:* {round(duration, 1)}s",
        f"• *Timestamp:* {time_str}"
    ]

    if near_miss_list:
        msg_lines.append("\n👀 *NEAR-MISS WATCHLIST (0.51% - 1.50%)*")
        msg_lines.append("------------------------------------")
        for t in near_miss_list:
            msg_lines.append(f"• {t['type']} *{t['pair']}* | Dist: {t['dist']}% | Price: {t['price']}")
    else:
        msg_lines.append("\n• _No tokens currently sitting in the 0.51%-1.50% near-miss zone._")

    send_telegram_alert("\n".join(msg_lines))

# ==========================================
# 2-HOUR HEARTBEAT & WATCHLIST COMPILER
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
        "\n👀 *NEAR-MISS WATCHLIST (Proximity 0.51% - 1.50%)*",
        "------------------------------------"
    ]

    if not watchlist_tokens:
        msg_lines.append("• _No near-miss tokens currently in the 0.51%-1.50% zone with RSI divergence._")
    else:
        for t in watchlist_tokens:
            msg_lines.append(
                f"• {t['type']} *{t['pair']}*\n"
                f"  - Distance: {t['dist']}% | Price: {t['price']} | 1H EMA: {t['ema_1h']}"
            )

    msg_lines.append("\n⚠️ *Keep an eye on these tokens as they approach your 0.50% entry zone!*")
    send_telegram_alert("\n".join(msg_lines))

# ==========================================
# BACKGROUND CONTINUOUS SCANNER THREAD
# ==========================================
def scanner_loop():
    global last_scan_time, last_markets_count, last_scan_duration, watchlist_tokens, last_primary_alerts_count

    # Send instant startup notification
    startup_msg = (
        "🚀 *CoinDCX Scanner Bot Started & Active!*\n\n"
        "• *Status:* 🟢 Online in Render Cloud\n"
        "• *Scan Rate:* Every 90 seconds\n"
        "• *Heartbeat Summary:* Every 2 hours"
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
                time.sleep(0.05)

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

# ==========================================
# FLASK ROUTES (KEEP-ALIVE & TEST LINK)
# ==========================================
@app.route('/')
def home():
    return f"🟢 CoinDCX Scanner Web Service Active! Last scan: {last_scan_time}", 200

@app.route('/test')
def trigger_test():
    threading.Thread(target=run_diagnostic_test).start()
    return "<h1>🟢 Manual Diagnostic Test Triggered! Check your Telegram app for the full report.</h1>", 200

# ==========================================
# START BACKGROUND SCANNER ON MODULE IMPORT
# ==========================================
scanner_thread = threading.Thread(target=scanner_loop, daemon=True)
scanner_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
