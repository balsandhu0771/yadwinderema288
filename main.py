import os
import time
import threading
import requests
import pandas as pd
from flask import Flask

# ==========================================
# CONFIGURATION
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

EMA_PERIOD = 288
RSI_PERIOD = 14
PROXIMITY_PRIMARY_PCT = 5.0    # Alert zone (<= 5.0% proximity or crossed)
PROXIMITY_WATCHLIST_MIN = 5.01 # Watchlist lower bound
PROXIMITY_WATCHLIST_MAX = 8.00 # Watchlist upper bound

DIV_LOOKBACK_MIN = 5           # Minimum swing lookback
DIV_LOOKBACK_MAX = 45          # Maximum swing lookback
MIN_CANDLES_REQUIRED = 80      # Minimum candle history
HEARTBEAT_INTERVAL = 7200      # 2 Hours in seconds

COINDCX_PUBLIC_API = "https://public.coindcx.com"
COINDCX_TICKER_API = "https://api.coindcx.com/exchange/ticker"
COINDCX_MARKETS_API = "https://api.coindcx.com/exchange/v1/markets"

app = Flask(__name__)

last_scan_time = "Not started"
last_markets_count = 0
last_scan_duration = 0
last_primary_alerts_count = 0
watchlist_tokens = []
scanner_started = False

# ==========================================
# TELEGRAM NOTIFIER
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
                print(f"[Error] Telegram chunk failed: {e}")

# ==========================================
# TECHNICAL ANALYSIS
# ==========================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def check_bearish_divergence(df: pd.DataFrame) -> bool:
    curr_idx = len(df) - 1
    curr_price = df['high'].iloc[curr_idx]
    curr_rsi = df['rsi'].iloc[curr_idx]

    start_idx = max(1, curr_idx - DIV_LOOKBACK_MAX)
    end_idx = max(1, curr_idx - DIV_LOOKBACK_MIN)

    for i in range(start_idx, end_idx):
        past_price = df['high'].iloc[i]
        past_rsi = df['rsi'].iloc[i]

        is_fractal_peak = (past_price >= df['high'].iloc[i - 1]) and (past_price >= df['high'].iloc[i + 1])
        if is_fractal_peak and (curr_price > past_price) and (curr_rsi < past_rsi):
            return True
    return False

def check_bullish_divergence(df: pd.DataFrame) -> bool:
    curr_idx = len(df) - 1
    curr_price = df['low'].iloc[curr_idx]
    curr_rsi = df['rsi'].iloc[curr_idx]

    start_idx = max(1, curr_idx - DIV_LOOKBACK_MAX)
    end_idx = max(1, curr_idx - DIV_LOOKBACK_MIN)

    for i in range(start_idx, end_idx):
        past_price = df['low'].iloc[i]
        past_rsi = df['rsi'].iloc[i]

        is_fractal_trough = (past_price <= df['low'].iloc[i - 1]) and (past_price <= df['low'].iloc[i + 1])
        if is_fractal_trough and (curr_price < past_price) and (curr_rsi > past_rsi):
            return True
    return False

# ==========================================
# PAIR RESOLVER & DATA FETCHER
# ==========================================
def get_all_pairs() -> list:
    """Fetches exact candle-compatible pair codes (e.g., B-BTC_USDT)."""
    try:
        res = requests.get(COINDCX_MARKETS_API, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                return [p for p in data if p.startswith("B-") or p.startswith("I-")]
    except Exception:
        pass

    # Fallback to ticker endpoint
    try:
        res = requests.get(COINDCX_TICKER_API, timeout=10)
        if res.status_code == 200:
            pairs = []
            for item in res.json():
                m = item.get('market', '')
                if m.endswith('USDT'):
                    pairs.append(f"B-{m[:-4]}_USDT")
                elif m.endswith('INR'):
                    pairs.append(f"I-{m[:-3]}_INR")
            return list(set(pairs))
    except Exception as e:
        print(f"[Error] Fetching markets: {e}")
    return []

def fetch_candles(pair: str, interval: str = "1h", limit: int = 350) -> pd.DataFrame:
    url = f"{COINDCX_PUBLIC_API}/market_data/candles"
    for _ in range(2):
        try:
            res = requests.get(url, params={"pair": pair, "interval": interval, "limit": limit}, timeout=10)
            if res.status_code == 200 and len(res.json()) > 0:
                df = pd.DataFrame(res.json())
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
                return df.iloc[::-1].reset_index(drop=True)
        except Exception:
            pass
        time.sleep(0.1)
    return pd.DataFrame()

# ==========================================
# 1-HOUR SCANNER LOGIC
# ==========================================
def process_market(pair: str, is_diagnostic: bool = False) -> dict:
    global watchlist_tokens

    df_1h = fetch_candles(pair, interval="1h")
    if df_1h.empty or len(df_1h) < MIN_CANDLES_REQUIRED:
        return {"status": "skipped"}

    df_1h['ema288'] = df_1h['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_1h['rsi'] = calculate_rsi(df_1h['close'], period=RSI_PERIOD)

    live_price = df_1h.iloc[-1]['close']
    ema_1h = df_1h.iloc[-1]['ema288']
    dist_pct = abs(live_price - ema_1h) / ema_1h * 100

    # Bearish Setup (1H Only)
    if check_bearish_divergence(df_1h):
        if dist_pct <= PROXIMITY_PRIMARY_PCT or live_price >= ema_1h:
            if not is_diagnostic:
                msg = (
                    f"🔴 *1H BEARISH EMA 288 ALERT*\n"
                    f"------------------------------------\n"
                    f"• *Market:* `{pair}`\n"
                    f"• *Live Price:* `{live_price}`\n"
                    f"• *1H 288 EMA:* `{round(ema_1h, 4)}`\n"
                    f"• *EMA Proximity:* `{round(dist_pct, 2)}%`\n"
                    f"• *Signal:* 1H Fractal Bearish RSI Divergence"
                )
                send_telegram_alert(msg)

            return {"status": "primary_alert", "pair": pair, "type": "🔴 Bearish", "price": live_price, "ema": round(ema_1h, 4), "dist": round(dist_pct, 2)}

        elif PROXIMITY_WATCHLIST_MIN <= dist_pct <= PROXIMITY_WATCHLIST_MAX:
            token_info = {"pair": pair, "type": "🔴 Bearish", "price": live_price, "ema": round(ema_1h, 4), "dist": round(dist_pct, 2)}
            if not is_diagnostic:
                watchlist_tokens.append(token_info)
            return {"status": "near_miss", "info": token_info}

    # Bullish Setup (1H Only)
    if check_bullish_divergence(df_1h):
        if dist_pct <= PROXIMITY_PRIMARY_PCT or live_price <= ema_1h:
            if not is_diagnostic:
                msg = (
                    f"🟢 *1H BULLISH EMA 288 ALERT*\n"
                    f"------------------------------------\n"
                    f"• *Market:* `{pair}`\n"
                    f"• *Live Price:* `{live_price}`\n"
                    f"• *1H 288 EMA:* `{round(ema_1h, 4)}`\n"
                    f"• *EMA Proximity:* `{round(dist_pct, 2)}%`\n"
                    f"• *Signal:* 1H Fractal Bullish RSI Divergence"
                )
                send_telegram_alert(msg)

            return {"status": "primary_alert", "pair": pair, "type": "🟢 Bullish", "price": live_price, "ema": round(ema_1h, 4), "dist": round(dist_pct, 2)}

        elif PROXIMITY_WATCHLIST_MIN <= dist_pct <= PROXIMITY_WATCHLIST_MAX:
            token_info = {"pair": pair, "type": "🟢 Bullish", "price": live_price, "ema": round(ema_1h, 4), "dist": round(dist_pct, 2)}
            if not is_diagnostic:
                watchlist_tokens.append(token_info)
            return {"status": "near_miss", "info": token_info}

    return {"status": "normal"}

# ==========================================
# DIAGNOSTIC ROUTE
# ==========================================
def run_diagnostic_test():
    start_time = time.time()
    markets = get_all_pairs()
    total_scanned = len(markets)
    primary_alerts_found = 0
    near_miss_list = []

    for pair in markets:
        res = process_market(pair, is_diagnostic=True)
        if res.get("status") == "primary_alert":
            primary_alerts_found += 1
        elif res.get("status") == "near_miss":
            near_miss_list.append(res["info"])
        time.sleep(0.04)

    duration = time.time() - start_time
    time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    msg_lines = [
        "🧪 *1H SCANNER DIAGNOSTIC REPORT*",
        "------------------------------------",
        f"• *Status:* 🟢 Active & Scanning",
        f"• *Valid Markets Checked:* {total_scanned}",
        f"• *Primary Alerts (<= 5.0%):* {primary_alerts_found}",
        f"• *Watchlist Tokens (5.01% - 8.0%):* {len(near_miss_list)}",
        f"• *Duration:* {round(duration, 1)}s",
        f"• *Timestamp:* {time_str}"
    ]

    if near_miss_list:
        msg_lines.append("\n👀 *NEAR-MISS WATCHLIST (5.01% - 8.00%)*")
        msg_lines.append("------------------------------------")
        for t in near_miss_list[:15]:
            msg_lines.append(f"• {t['type']} *{t['pair']}* | Dist: {t['dist']}% | Price: {t['price']}")
    else:
        msg_lines.append("\n• _No tokens currently in the 5.01%-8.00% watchlist window._")

    send_telegram_alert("\n".join(msg_lines))

# ==========================================
# 2-HOUR HEARTBEAT
# ==========================================
def send_2hour_heartbeat():
    global watchlist_tokens, last_primary_alerts_count
    
    msg_lines = [
        "💓 *2-HOUR SYSTEM HEARTBEAT*",
        "------------------------------------",
        f"• *Status:* 🟢 Bot Online",
        f"• *Pairs Scanned:* {last_markets_count}",
        f"• *Alerts Triggered (Last Cycle):* {last_primary_alerts_count}",
        f"• *Last Completed Scan:* {last_scan_time}",
        "\n👀 *WATCHLIST DIGEST (Proximity 5.01% - 8.00%)*",
        "------------------------------------"
    ]

    if not watchlist_tokens:
        msg_lines.append("• _No near-miss tokens currently logged._")
    else:
        for t in watchlist_tokens[:12]:
            msg_lines.append(f"• {t['type']} *{t['pair']}* | Dist: {t['dist']}% | Price: {t['price']}")

    send_telegram_alert("\n".join(msg_lines))

# ==========================================
# SCANNER LOOP
# ==========================================
def scanner_loop():
    global last_scan_time, last_markets_count, last_scan_duration, watchlist_tokens, last_primary_alerts_count

    startup_msg = (
        "🚀 *CoinDCX 1-Hour Scanner Online!*\n\n"
        "• *Timeframe:* 1-Hour Exclusive\n"
        "• *Proximity Zone:* <= 5.0%\n"
        "• *Watchlist Zone:* 5.01% - 8.00%\n"
        "• *Pattern:* Fractal RSI Divergence (5–45 Candles)"
    )
    send_telegram_alert(startup_msg)

    last_heartbeat_time = time.time()

    while True:
        try:
            start_time = time.time()
            watchlist_tokens = []
            cycle_alerts = 0

            markets = get_all_pairs()
            last_markets_count = len(markets)

            for pair in markets:
                res = process_market(pair)
                if res.get("status") == "primary_alert":
                    cycle_alerts += 1
                time.sleep(0.04)

            last_primary_alerts_count = cycle_alerts
            last_scan_duration = time.time() - start_time
            last_scan_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            print(f"[Scan Complete] Scanned {last_markets_count} pairs in {round(last_scan_duration, 1)}s | Alerts: {last_primary_alerts_count}")

            if time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                send_2hour_heartbeat()
                last_heartbeat_time = time.time()

            time.sleep(90)

        except Exception as e:
            print(f"[Loop Error]: {e}")
            time.sleep(60)

def start_scanner_once():
    global scanner_started
    if not scanner_started:
        scanner_started = True
        t = threading.Thread(target=scanner_loop, daemon=True)
        t.start()

# ==========================================
# FLASK WEB ROUTES
# ==========================================
@app.route('/')
def home():
    start_scanner_once()
    return f"🟢 1-Hour Scanner Active! Last scan: {last_scan_time}", 200

@app.route('/test')
def trigger_test():
    start_scanner_once()
    threading.Thread(target=run_diagnostic_test).start()
    return "<h1>🟢 Manual Diagnostic Test Triggered! Check Telegram.</h1>", 200

start_scanner_once()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
