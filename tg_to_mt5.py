"""
Telegram -> MT5 bridge (IC Markets / XM)
- Reads config from .env and config.yaml
- Parses GOLD buy/sell zone signals (both formats you shared)
- Places market or limit orders with per-TP targets + shared SL
- Optional martingale layer
- Moves remaining positions to BE after TP1

Folder layout:
  tg_to_mt5.py
  .env
  config.yaml
  logs/bridge.log
  services/install_nssm.bat
  get_chat_id.py
"""

import os, re, time, math, hashlib, logging, sys
from typing import Optional, Dict, Any, List

import yaml
from dotenv import load_dotenv
from telethon import TelegramClient, events
import MetaTrader5 as mt5

# --------------------------
# Load .env and config.yaml
# --------------------------
load_dotenv()

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise SystemExit("config.yaml not found. Create it next to this script.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

# --------------------------
# Read settings (env > yaml)
# --------------------------
def cfg(path, default=None):
    cur = CFG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

API_ID = int(os.getenv("TG_API_ID", "0"))
API_HASH = os.getenv("TG_API_HASH", "")
SESSION_NAME = os.getenv("TG_SESSION_NAME", cfg("telegram.session_name", "tg_mt5_session"))
ALLOWED_CHAT_ID = os.getenv("ALLOWED_CHAT_ID", cfg("guards.allowlist_chat_id"))
ALLOWED_CHAT_IDS = cfg("guards.allowlist_chat_ids", [])
if ALLOWED_CHAT_ID:
    ALLOWED_CHAT_IDS.append(ALLOWED_CHAT_ID)
ALLOWED_CHAT_IDS = [int(x) for x in ALLOWED_CHAT_IDS if str(x).strip()]

# Risk / trade behavior
RISK_PER_TRADE     = float(os.getenv("RISK_PER_TRADE", str(cfg("risk.risk_per_trade", 0.01))))
MAX_ORDERS         = int(os.getenv("MAX_ORDERS", str(cfg("risk.max_orders", 4))))
SLIPPAGE_POINTS    = int(os.getenv("SLIPPAGE_POINTS", str(cfg("risk.slippage_points", 30))))
BE_BUFFER_POINTS   = int(os.getenv("BE_BUFFER_POINTS", str(cfg("risk.be_buffer_points", 10))))
KILL_SWITCH_DD_PCT = float(os.getenv("KILL_SWITCH_DD_PCT", str(cfg("risk.kill_switch_dd_pct", 0))))

# Symbols
GOLD_SYMBOL_CANDIDATES = cfg("symbol_map.GOLD.candidates",
                             ["XAUUSD", "XAUUSD.i", "XAUUSD.m", "XAUUSD.x", "GOLD", "GOLDmicro"])

# Regex (from config or defaults)
ZONE_SAME_LINE = cfg("parsing.zone_same_line_regex",
                     r"(?i)\b(buy|sell)\b.*?\bzone\b.*?([0-9.,]+)\s*[-–—]\s*([0-9.,]+)")
ZONE_NEXT_LINE = cfg("parsing.zone_next_line_regex",
                     r"(?m)^\s*([0-9.,]+)\s*[-–—]\s*([0-9.,]+)\s*$")
TP_REGEX       = cfg("parsing.tp_regex", r"(?i)TP\s*\d+\s*[-:]\s*([0-9.,]+)")
SL_REGEX       = cfg("parsing.sl_regex", r"(?i)\bSL\b\s*[-:]\s*([0-9.,]+)")
SIDE_REGEX     = cfg("parsing.side_regex", r"(?i)\b(buy|sell)\b")
MART_REGEX     = cfg("parsing.martingale_regex", r"(?i)\bmartingale\b")
REQUIRE_HINT   = cfg("parsing.require_symbol_hint", "GOLD")

ZONE_RE_A = re.compile(ZONE_SAME_LINE)
ZONE_RE_B = re.compile(ZONE_NEXT_LINE)
TP_RE     = re.compile(TP_REGEX)
SL_RE     = re.compile(SL_REGEX)
SIDE_RE   = re.compile(SIDE_REGEX)
MART_RE   = re.compile(MART_REGEX)

# --------------------------
# Logging (console + file)
# --------------------------
LOG_DIR = os.path.join("logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "bridge.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ],
)
log = logging.getLogger("tg-mt5")

# --------------------------
# Telegram client
# --------------------------
if not API_ID or not API_HASH:
    raise SystemExit("Set TG_API_ID and TG_API_HASH in .env or environment variables.")

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

# --------------------------
# Parsing
# --------------------------
def _to_float(s: str) -> float:
    return float(s.replace(",", "").strip())

def parse_signal(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()

    if REQUIRE_HINT and REQUIRE_HINT.lower() not in raw.lower():
        return None

    m_side = SIDE_RE.search(raw)
    if not m_side:
        return None
    side = m_side.group(1).upper()

    m_zone = ZONE_RE_A.search(raw)
    if m_zone:
        z1, z2 = _to_float(m_zone.group(2)), _to_float(m_zone.group(3))
    else:
        m_zone2 = ZONE_RE_B.search(raw)
        if not m_zone2:
            return None
        z1, z2 = _to_float(m_zone2.group(1)), _to_float(m_zone2.group(2))
    z_lo, z_hi = sorted([z1, z2])

    tps = [_to_float(x) for x in TP_RE.findall(raw)]
    slm = SL_RE.search(raw)
    sl  = _to_float(slm.group(1)) if slm else None
    martingale = bool(MART_RE.search(raw))

    return {
        "symbol_hint": REQUIRE_HINT or "GOLD",
        "side": side,
        "zone_low": z_lo,
        "zone_high": z_hi,
        "tps": tps[:4],
        "sl": sl,
        "martingale": martingale
    }

# --------------------------
# MT5 helpers
# --------------------------
def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    ai = mt5.account_info()
    if ai is None:
        raise RuntimeError("Open MT5 terminal and log into your account.")
    log.info(f"MT5 connected: {ai.login} @ {ai.company}")

def pick_symbol(cands: List[str]) -> str:
    for s in cands:
        info = mt5.symbol_info(s)
        if info and (info.visible or mt5.symbol_select(s, True)):
            return s
    raise RuntimeError("Could not select GOLD symbol. Add your exact broker symbol to config.yaml.")

def get_tick(symbol: str):
    t = mt5.symbol_info_tick(symbol)
    if t is None:
        raise RuntimeError(f"No tick for {symbol}")
    return t

def points(symbol: str):
    info = mt5.symbol_info(symbol)
    if not info:
        raise RuntimeError(f"No symbol info for {symbol}")
    return info.point, info.digits

def account_balance() -> float:
    ai = mt5.account_info()
    return ai.balance if ai else 0.0

def daily_drawdown_exceeded(limit_pct: float) -> bool:
    if limit_pct <= 0:
        return False
    ai = mt5.account_info()
    if not ai or ai.balance <= 0:
        return False
    dd_pct = (ai.equity - ai.balance) / ai.balance * 100.0
    return dd_pct <= -abs(limit_pct)

def calc_lot_from_risk(symbol: str, entry: float, sl_price: Optional[float], risk_frac: float) -> float:
    si = mt5.symbol_info(symbol)
    if not si or not sl_price:
        return max(si.volume_min if si else 0.1, 0.1)
    tick_val  = si.trade_tick_value
    tick_size = si.trade_tick_size if si.trade_tick_size > 0 else si.point
    price_diff = abs(entry - sl_price)
    ticks = price_diff / (tick_size if tick_size > 0 else si.point)
    risk_amount = account_balance() * risk_frac
    lots = 0.10 if (tick_val <= 0 or ticks <= 0) else (risk_amount / (ticks * tick_val))
    # clamp & step
    lots = max(si.volume_min, min(si.volume_max, lots))
    step = si.volume_step if si.volume_step > 0 else 0.01
    lots = math.floor(lots / step) * step
    return round(lots, 2)

def order_send(symbol: str, side_buy: bool, lot: float, price: Optional[float]=None,
               sl: float=0.0, tp: float=0.0, pending: bool=False, magic: int=946031):
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Cannot select {symbol}")

    if pending:
        otype  = mt5.ORDER_TYPE_BUY_LIMIT if side_buy else mt5.ORDER_TYPE_SELL_LIMIT
        action = mt5.TRADE_ACTION_PENDING
        px     = price
    else:
        otype  = mt5.ORDER_TYPE_BUY if side_buy else mt5.ORDER_TYPE_SELL
        action = mt5.TRADE_ACTION_DEAL
        tick   = get_tick(symbol)
        px     = tick.ask if side_buy else tick.bid

    req = {
        "action": action,
        "symbol": symbol,
        "volume": lot,
        "type": otype,
        "price": px,
        "sl": sl,
        "tp": tp,
        "deviation": SLIPPAGE_POINTS,
        "magic": magic,
        "comment": "tg-bridge",
        "type_filling": mt5.ORDER_FILLING_IOC if not pending else mt5.ORDER_FILLING_RETURN,
        "type_time": mt5.ORDER_TIME_GTC
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"order_send failed: {getattr(res, 'retcode', None)} {getattr(res, 'comment', '')}")
    return res

def move_to_be(symbol: str, magic: int, be_buffer_pts: int=0):
    pt, digits = points(symbol)
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    for p in positions:
        if p.magic != magic:
            continue
        is_buy = (p.type == mt5.POSITION_TYPE_BUY)
        be = p.price_open + (be_buffer_pts * pt if is_buy else -be_buffer_pts * pt)
        need = (p.sl is None) or (is_buy and p.sl < be) or ((not is_buy) and p.sl > be)
        if need:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "sl": round(be, digits),
                "tp": p.tp
            })

def maybe_add_martingale_layer(symbol: str, side_buy: bool, z_lo: float, z_hi: float,
                               lot: float, sl: float, digits: int):
    # One extra limit order at zone edge (use smaller lot if you prefer)
    edge = z_lo if side_buy else z_hi
    try:
        order_send(symbol, side_buy, lot, price=round(edge, digits), sl=sl, tp=0.0, pending=True)
        log.info(f"Martingale layer placed @ {edge}")
    except Exception as e:
        log.warning(f"Martingale layer failed: {e}")

# --------------------------
# Trade execution
# --------------------------
def execute_trade(sig: Dict[str, Any], magic: int=946031):
    symbol = pick_symbol(GOLD_SYMBOL_CANDIDATES)
    pt, digits = points(symbol)
    tick = get_tick(symbol)
    cur_ask, cur_bid = tick.ask, tick.bid

    side = sig["side"]
    z_lo = sig["zone_low"]
    z_hi = sig["zone_high"]
    sl   = sig["sl"]
    tps  = sig["tps"] or []

    mid = (z_lo + z_hi) / 2.0
    side_buy = (side == "BUY")
    cur = cur_ask if side_buy else cur_bid
    inside = (z_lo <= cur <= z_hi)

    # Risk sizing
    lot = calc_lot_from_risk(symbol, mid, sl, RISK_PER_TRADE) if sl else 0.10

    # Number of orders
    n = min(len(tps), MAX_ORDERS) if tps else 1

    # Build entries
    if inside:
        entries = [("market", cur)] * n
    else:
        if side_buy:
            levels = [mid, z_lo]
        else:
            levels = [mid, z_hi]
        while len(levels) < n:
            levels.append(levels[-1])
        entries = [("limit", round(p, digits)) for p in levels[:n]]

    sl_p = round(sl, digits) if sl else 0.0

    # Place orders
    for i in range(n):
        tp = round(tps[i], digits) if i < len(tps) else 0.0
        typ, price = entries[i]
        res = order_send(symbol, side_buy, lot,
                         price=(price if typ == "limit" else None),
                         sl=sl_p, tp=tp, pending=(typ == "limit"), magic=magic)
        log.info(f"{side} {symbol} -> ticket {getattr(res, 'order', None)} | {typ} | TP:{tp} SL:{sl_p}")

    log.info("Orders placed. BE rule active: when TP1 hits, remaining will move to BE.")

    if sig.get("martingale"):
        maybe_add_martingale_layer(symbol, side_buy, z_lo, z_hi, lot, sl_p, digits)

# --------------------------
# Telegram handler
# --------------------------
SEEN = set()

@client.on(events.NewMessage(chats=ALLOWED_CHAT_IDS if ALLOWED_CHAT_IDS else None))
async def on_msg(event):
    if KILL_SWITCH_DD_PCT > 0 and daily_drawdown_exceeded(KILL_SWITCH_DD_PCT):
        log.warning("Kill-switch active (daily DD exceeded). Skipping trades.")
        return

    txt = (event.raw_text or "").replace("\u200b", "")
    fid = hashlib.sha1((txt or "").encode("utf-8")).hexdigest()
    if fid in SEEN:
        return
    SEEN.add(fid)

    first = txt.splitlines()[0] if txt else ""
    log.info(f"[IN] chat={event.chat_id} | {first!r}")

    sig = parse_signal(txt)
    if not sig:
        log.info("[SKIP] format not matched; waiting for GOLD + zone + TP/SL")
        return

    try:
        connect_mt5()
        execute_trade(sig)
        # quick BE watchdog
        symbol = pick_symbol(GOLD_SYMBOL_CANDIDATES)
        move_to_be(symbol, magic=946031, be_buffer_pts=BE_BUFFER_POINTS)
    except Exception as e:
        log.error(f"[ERROR] {e}")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    if not ALLOWED_CHAT_IDS:
        log.warning("ALLOWED_CHAT_ID(S) not set. The bot will listen to ALL chats in this account.")

    log.info("Listening for GOLD signals…")
    client.start()
    client.run_until_disconnected()
