"""
Two-step Telegram -> MT5 bridge

Step 1: "GOLD buy now" / "GOLD sell now"
    -> place market order(s) immediately (no SL/TP yet)

Step 2: "GOLD ... in this zone ... TP... SL ..."
    -> modify existing GOLD positions (same magic) to set SL
       and assign TP1..TPn across them (1 TP per position).
       If only one position exists, it gets TP1.

Optional: if the update text includes "hit TP1, set BE", we
enable BE (break-even) behavior when TP1 hits.

Files:
  - .env                  (API keys etc.)
  - config.yaml           (symbol list, risk, regex)
  - logs/bridge.log
"""

import os, re, time, math, hashlib, logging, sys
from typing import Optional, Dict, Any, List, Tuple

import yaml
from dotenv import load_dotenv
from telethon import TelegramClient, events
import MetaTrader5 as mt5
import argparse
import asyncio
# --------------------------
# Load .env and config.yaml
# --------------------------
load_dotenv()

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise SystemExit("config.yaml not found. Create it next to this script.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

def cfg(path, default=None):
    cur = CFG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# --------------------------
# Settings (env > yaml)
# --------------------------
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

# "Now" behavior (new)
NOW_ORDERS   = int(os.getenv("NOW_ORDERS", "1"))        # how many positions to open on "buy/sell now"
NOW_DEFAULT_LOT = float(os.getenv("NOW_DEFAULT_LOT", "0.10"))  # lot for each "now" position (no SL yet)

# Symbols
GOLD_SYMBOL_CANDIDATES = cfg("symbol_map.GOLD.candidates",
                             ["XAUUSD", "XAUUSD.i", "XAUUSD.m", "XAUUSD.x", "GOLD", "GOLDmicro"])

# How many entry levels inside the zone (per direction)
ZONE_ORDERS = int(os.getenv("ZONE_ORDERS", "2"))          # e.g., 2 levels (mid + high/low)
# How many TPs for each entry
TPS_PER_ORDER = int(os.getenv("TPS_PER_ORDER", "3"))      # TP1..TP3

# Lot size per entry (will be split between children)
ZONE_TOTAL_LOT = float(os.getenv("ZONE_TOTAL_LOT", "0.15"))  # e.g., 0.15 lot per entry
# Martingale multiplier
MARTINGALE_FACTOR = float(os.getenv("MARTINGALE_FACTOR", "1.5"))  # 1.5x bigger lots on martingale
# Buffer for break-even (in points)
BE_BUFFER_POINTS = int(os.getenv("BE_BUFFER_POINTS", "10"))
MAGIC = 946031

# --------------------------
# Logging
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
# Telegram
# --------------------------
if not API_ID or not API_HASH:
    raise SystemExit("Set TG_API_ID and TG_API_HASH in .env or environment variables.")

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

# --------------------------
# Parsing (NOW + UPDATE)
# --------------------------
# "NOW" signals: GOLD buy now / GOLD sell now (case-insensitive, tolerate extra words)
NOW_RE = re.compile(r'(?i)\bgold\b.*\b(buy|sell)\b.*\bnow\b')

# UPDATE signals: zone + TP/SL (various formats)
ZONE_SAME_LINE = cfg("parsing.zone_same_line_regex",
                     r"(?i)\b(buy|sell)\b.*?\bzone\b.*?([0-9.,]+)\s*[-–—]\s*([0-9.,]+)")
ZONE_NEXT_LINE = cfg("parsing.zone_next_line_regex",
                     r"(?m)^\s*([0-9.,]+)\s*[-–—]\s*([0-9.,]+)\s*$")
TP_REGEX       = cfg("parsing.tp_regex", r"(?i)TP\s*[-]?\s*\d*\s*[-:]\s*([0-9.,]+)")  # tolerate "TP- 3341.00"
SL_REGEX       = cfg("parsing.sl_regex", r"(?i)\bSL\b\s*[-:]\s*([0-9.,]+)")
SIDE_REGEX     = cfg("parsing.side_regex", r"(?i)\b(buy|sell)\b")
MART_REGEX     = cfg("parsing.martingale_regex", r"(?i)\bmartingale\b")

ZONE_RE_A = re.compile(ZONE_SAME_LINE)
ZONE_RE_B = re.compile(ZONE_NEXT_LINE, re.MULTILINE)
TP_RE     = re.compile(TP_REGEX)
SL_RE     = re.compile(SL_REGEX)
SIDE_RE   = re.compile(SIDE_REGEX)
MART_RE   = re.compile(MART_REGEX)
BE_HINT_RE= re.compile(r'(?i)hit\s*TP1.*set\s*BE')

def _to_float(s: str) -> float:
    return float(s.replace(",", "").strip())

def parse_message(text: str) -> Optional[Dict[str, Any]]:
    """
    Returns:
      {"type":"NOW","side":"BUY"/"SELL"}  OR
      {"type":"UPDATE","side":..,"zone_low":..,"zone_high":..,"tps":[..],"sl":..,"martingale":bool,"be_after_tp1":bool}
    """
    if not text:
        return None
    raw = text.strip()

    # NOW?
    m_now = NOW_RE.search(raw)
    if m_now:
        side = m_now.group(1).upper()
        return {"type":"NOW", "side":side}

    # UPDATE?
    if "gold" not in raw.lower():
        return None
    m_side = SIDE_RE.search(raw)
    if not m_side:
        return None
    side = m_side.group(1).upper()

    z1 = z2 = None
    m_zone = ZONE_RE_A.search(raw)
    if m_zone:
        z1, z2 = _to_float(m_zone.group(2)), _to_float(m_zone.group(3))
    else:
        m_zone2 = ZONE_RE_B.search(raw)
        if m_zone2:
            z1, z2 = _to_float(m_zone2.group(1)), _to_float(m_zone2.group(2))
    if z1 is None or z2 is None:
        return None

    z_lo, z_hi = sorted([z1, z2])
    tps = [_to_float(x) for x in TP_RE.findall(raw)]
    slm = SL_RE.search(raw)
    sl  = _to_float(slm.group(1)) if slm else None
    martingale = bool(MART_RE.search(raw))
    be_after_tp1 = bool(BE_HINT_RE.search(raw))

    return {
        "type":"UPDATE",
        "side": side,
        "zone_low": z_lo,
        "zone_high": z_hi,
        "tps": tps[:4],
        "sl": sl,
        "martingale": martingale,
        "be_after_tp1": be_after_tp1
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

def points(symbol: str) -> Tuple[float,int]:
    info = mt5.symbol_info(symbol)
    if not info:
        raise RuntimeError(f"No symbol info for {symbol}")
    return info.point, info.digits

def order_send_market(symbol: str, is_buy: bool, lot: float, sl: float=0.0, tp: float=0.0, magic: int=MAGIC):
    tick = get_tick(symbol)
    px = tick.ask if is_buy else tick.bid
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
        "price": px,
        "sl": sl, "tp": tp,
        "deviation": SLIPPAGE_POINTS,
        "magic": magic,
        "comment": "tg-bridge-now",
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"market send failed: {getattr(res,'retcode',None)} {getattr(res,'comment','')}")
    return res

def set_position_sltp_safe(symbol: str, pos, sl_req: float | None = None, tp_req: float | None = None):
    """
    Modify a position's SL/TP safely.
    - Either sl_req or tp_req (or both) may be None.
    - If None, we keep the current value on the position.
    """
    # use current values if not provided
    cur_sl = float(pos.sl) if getattr(pos, "sl", 0.0) else 0.0
    cur_tp = float(pos.tp) if getattr(pos, "tp", 0.0) else 0.0
    sl_in  = cur_sl if (sl_req is None) else sl_req
    tp_in  = cur_tp if (tp_req is None) else tp_req

    sl_ok, tp_ok = bounded_sltp_for_position(symbol, pos, sl_in, tp_in)
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "sl": sl_ok,
        "tp": tp_ok
    }
    r = mt5.order_send(req)
    if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(
            f"modify failed ticket {pos.ticket}: {getattr(r,'retcode',None)} {getattr(r,'comment','')} "
            f"(requested SL/TP {sl_in}/{tp_in} -> adjusted {sl_ok}/{tp_ok})"
        )

def get_open_positions(symbol: str, magic: int=MAGIC):
    poss = mt5.positions_get(symbol=symbol)
    if not poss: return []
    return [p for p in poss if p.magic == magic]

def move_to_be_all(symbol: str, be_buffer_pts: int=0, magic: int=MAGIC):
    pt, digits = points(symbol)
    for p in get_open_positions(symbol, magic):
        is_buy = (p.type == mt5.POSITION_TYPE_BUY)
        be = p.price_open + (be_buffer_pts * pt if is_buy else -be_buffer_pts * pt)
        need = (p.sl is None) or (is_buy and p.sl < be) or ((not is_buy) and p.sl > be)
        if need:
            set_position_sltp_safe(symbol, p, round(be, digits), p.tp)

def snap_to_tick(symbol: str, price: float) -> float:
    si = mt5.symbol_info(symbol)
    if not si:
        return price
    step = si.trade_tick_size if si.trade_tick_size > 0 else si.point
    if step <= 0:
        return round(price, si.digits)
    # snap down/up to nearest tick
    ticks = round(price / step)
    snapped = ticks * step
    return round(snapped, si.digits)

def sltp_constraints(symbol: str):
    si = mt5.symbol_info(symbol)
    if not si:
        return (0.0, 0, 0, 0)

    pt = si.point
    digits = si.digits

    # Some terminals expose trade_stops_level / trade_freeze_level only
    # Fall back to older names if present; default 0 if missing.
    stops_level = getattr(si, "trade_stops_level", 0) or getattr(si, "stops_level", 0) or 0
    freeze_level = getattr(si, "trade_freeze_level", 0) or getattr(si, "freeze_level", 0) or 0

    # return points-per-level (raw levels are in "points" units, not price)
    return pt, digits, int(stops_level), int(freeze_level)


def bounded_sltp_for_position(symbol: str, pos, sl_req: float | None, tp_req: float | None) -> tuple[float, float]:
    """
    Returns (sl_ok, tp_ok) adjusted to satisfy:
      - correct side (SL below for BUY, above for SELL; TP the opposite)
      - min distance: stops_level from Bid/Ask
      - not in freeze_level
      - snapped to tick size
    """
    tick = mt5.symbol_info_tick(symbol)
    pt, digits, stops_level, freeze_level = sltp_constraints(symbol)
    ask, bid = tick.ask, tick.bid
    min_dist = (stops_level or 0) * pt
    freeze_dist = (freeze_level or 0) * pt

    is_buy = (pos.type == mt5.POSITION_TYPE_BUY)

    # Start from requested values
    sl = sl_req
    tp = tp_req

    if is_buy:
        # enforce sides
        if sl is not None and sl > 0:
            # SL must be BELOW Bid by at least min_dist + freeze_dist
            max_sl = bid - max(min_dist, freeze_dist)
            sl = min(sl, max_sl)
        if tp is not None and tp > 0:
            # TP must be ABOVE Bid by at least min_dist + freeze_dist
            min_tp = bid + max(min_dist, freeze_dist)
            tp = max(tp, min_tp)
    else:
        if sl is not None and sl > 0:
            # SL must be ABOVE Ask by at least min_dist + freeze_dist
            min_sl = ask + max(min_dist, freeze_dist)
            sl = max(sl, min_sl)
        if tp is not None and tp > 0:
            # TP must be BELOW Ask by at least min_dist + freeze_dist
            max_tp = ask - max(min_dist, freeze_dist)
            tp = min(tp, max_tp)

    # snap to tick size & digits
    sl = snap_to_tick(symbol, sl) if sl and sl > 0 else 0.0
    tp = snap_to_tick(symbol, tp) if tp and tp > 0 else 0.0

    return (round(sl, digits), round(tp, digits))

# ---------- ACTIVE TRADES HELPERS ----------

def positions_all(symbol: str | None = None, magic: int | None = None):
    """Return a list of mt5 positions, optionally filtered by symbol and magic."""
    poss = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if poss is None:
        return []
    if magic is None:
        return list(poss)
    return [p for p in poss if getattr(p, "magic", None) == magic]

def positions_for_gold(magic: int) -> list:
    """Convenience: get open GOLD positions for the script's MAGIC across candidate symbols."""
    for sym in GOLD_SYMBOL_CANDIDATES:
        info = mt5.symbol_info(sym)
        if info and (info.visible or mt5.symbol_select(sym, True)):
            poss = positions_all(symbol=sym, magic=magic)
            if poss:
                return poss
    return []

def fmt_pos(p) -> str:
    side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
    return (f"ticket={p.ticket} | {p.symbol} | {side} | lot={p.volume} | "
            f"open={p.price_open} | sl={p.sl or 0} | tp={p.tp or 0} | "
            f"profit={p.profit:.2f}")

def print_positions(symbol: str | None = None, magic: int | None = None):
    """Log current open positions (pretty)."""
    poss = positions_all(symbol=symbol, magic=magic)
    if not poss:
        log.info("No open positions." + (f" (symbol={symbol})" if symbol else ""))
        return
    log.info(f"Open positions: {len(poss)}" + (f" (symbol={symbol})" if symbol else ""))
    for p in sorted(poss, key=lambda x: x.time):
        log.info("  " + fmt_pos(p))

def entry_levels_for_zone(side: str, z_lo: float, z_hi: float, n: int) -> list[float]:
    """Return n entry levels inside the zone. SELL uses [mid, high], BUY uses [mid, low]."""
    mid = (z_lo + z_hi) / 2.0
    if side == "SELL":
        base = [mid, z_hi]
    else:
        base = [mid, z_lo]
    # pad if n>2
    while len(base) < n:
        base.append(base[-1])
    return base[:n]

def place_zone_orders(symbol: str, side: str, z_lo: float, z_hi: float,
                      tps: list[float], sl: float, n_entries: int,
                      children_per_entry: int, total_lot_per_entry: float,
                      martingale: bool):
    """
    For each entry level, place 'children_per_entry' pending orders with same SL and different TPs.
    Example: SELL zone -> SELL_LIMIT at each entry level.
    """
    si = mt5.symbol_info(symbol)
    pt, digits = points(symbol)
    side_buy = (side == "BUY")

    # choose TP ordering (SELL: high->low, BUY: low->high)
    if side == "BUY":
        tps_sorted = sorted(tps)[:children_per_entry]
    else:
        tps_sorted = sorted(tps, reverse=True)[:children_per_entry]

    if not tps_sorted:
        log.warning("No TPs provided for zone orders; skipping.")
        return

    levels = entry_levels_for_zone(side, z_lo, z_hi, n_entries)
    child_lot = round(total_lot_per_entry / max(1, children_per_entry), 2)

    group_id = int(time.time())  # simple sibling tag

    for lvl in levels:
        for i, tp in enumerate(tps_sorted, start=1):
            req_price = round(lvl, digits)
            req_sl = round(sl, digits)
            req_tp = round(tp, digits)
            # pending order type
            otype = mt5.ORDER_TYPE_BUY_LIMIT if side_buy else mt5.ORDER_TYPE_SELL_LIMIT
            req = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": child_lot,
                "type": otype,
                "price": req_price,
                "sl": req_sl,
                "tp": req_tp,
                "deviation": SLIPPAGE_POINTS,
                "magic": MAGIC,
                "comment": f"tg-zone grp:{group_id} L:{req_price} TP{i}",
                "type_filling": mt5.ORDER_FILLING_RETURN,
                "type_time": mt5.ORDER_TIME_GTC
            }
            res = mt5.order_send(req)
            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                raise RuntimeError(f"zone order failed: {getattr(res,'retcode',None)} {getattr(res,'comment','')}")
            log.info(f"Placed {side} LIMIT @{req_price} lot={child_lot} SL={req_sl} TP={req_tp} (grp {group_id})")

    # optional martingale: one extra at the zone edge (more conservative: use smaller step or same child_lot * factor)
    if martingale:
        edge = z_hi if not side_buy else z_lo
        lvl = round(edge, digits)
        # Create children at the edge too:
        m_lot = round(child_lot * MARTINGALE_FACTOR, 2)
        for i, tp in enumerate(tps_sorted, start=1):
            req = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": m_lot,
                "type": (mt5.ORDER_TYPE_BUY_LIMIT if side_buy else mt5.ORDER_TYPE_SELL_LIMIT),
                "price": lvl,
                "sl": round(sl, digits),
                "tp": round(tp, digits),
                "deviation": SLIPPAGE_POINTS,
                "magic": MAGIC,
                "comment": f"tg-zone-marti grp:{group_id} L:{lvl} TP{i}",
                "type_filling": mt5.ORDER_FILLING_RETURN,
                "type_time": mt5.ORDER_TIME_GTC
            }
            res = mt5.order_send(req)
            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                log.warning(f"martingale child failed: {getattr(res,'retcode',None)} {getattr(res,'comment','')}")
        log.info(f"Martingale layer placed at {lvl} (x{MARTINGALE_FACTOR}).")

# BE watcher (trigger BE when price reaches TP1)
LAST_BE_RULE = {"enabled": False, "side": None, "tp1": None, "symbol": None}

async def be_price_watcher():
    """Poll price every ~2s; if price crosses TP1 in the profitable direction, move SL to BE for siblings."""
    while True:
        try:
            if LAST_BE_RULE["enabled"] and LAST_BE_RULE["symbol"] and LAST_BE_RULE["tp1"] is not None:
                sym = LAST_BE_RULE["symbol"]
                side = LAST_BE_RULE["side"]
                tick = mt5.symbol_info_tick(sym)
                bid, ask = tick.bid, tick.ask
                tp1 = LAST_BE_RULE["tp1"]
                triggered = False
                if side == "SELL" and bid <= tp1:
                    triggered = True
                if side == "BUY"  and ask >= tp1:
                    triggered = True
                if triggered:
                    move_to_be_all(sym, be_buffer_pts=BE_BUFFER_POINTS, magic=MAGIC)
                    log.info("[BE] TP1 touched -> moved SLs to BE")
                    LAST_BE_RULE["enabled"] = False  # one-shot
        except Exception as e:
            log.warning(f"be_price_watcher error: {e}")
        await asyncio.sleep(2)

# --------------------------
# NEW: Execute NOW & UPDATE
# --------------------------
def handle_now(side: str):
    symbol = pick_symbol(GOLD_SYMBOL_CANDIDATES)
    is_buy = (side == "BUY")
    for i in range(max(1, NOW_ORDERS)):
        res = order_send_market(symbol, is_buy, NOW_DEFAULT_LOT)
        log.info(f"{side} NOW {symbol} -> ticket {getattr(res,'order',None)} lot {NOW_DEFAULT_LOT}")
    log.info("Placed NOW market order(s). Waiting for UPDATE to set SL/TPs.")

def handle_update(sig: Dict[str,Any]):
    symbol = pick_symbol(GOLD_SYMBOL_CANDIDATES)
    pt, digits = points(symbol)
    side = sig["side"]
    tps  = sig.get("tps", [])
    sl   = sig.get("sl", None)
    be_after_tp1 = sig.get("be_after_tp1", False)
    martingale = sig.get("martingale", False)

    # 1) Modify any existing NOW positions (set common SL, and optionally set TP1 to them)
    positions = get_open_positions(symbol, MAGIC)
    if positions:
        positions.sort(key=lambda p: p.time)
        common_sl = round(sl, digits) if sl is not None else 0.0
        # Give NOW positions TP1 if available, else leave TP as-is
        if tps:
            # choose first TP in correct direction
            tp1 = (sorted(tps)[0] if side=="BUY" else sorted(tps, reverse=True)[0])
            tp1 = round(tp1, digits)
        else:
            tp1 = None

        for p in positions:
            set_position_sltp_safe(symbol, p, common_sl, tp1)
            log.info(f"Modified NOW ticket {p.ticket}: SL {common_sl} | TP {tp1 if tp1 else p.tp or 0.0}")
        # tiny delay so server reflects new SL/TPs
        time.sleep(0.15)

    # 2) Place zone orders (2 entries × 3 TPs each)
    z_lo, z_hi = sig["zone_low"], sig["zone_high"]
    if sl is None or not tps:
        log.warning("Zone update missing SL or TPs; skipping placement of zone orders.")
    else:
        place_zone_orders(symbol, side, z_lo, z_hi, tps, sl,
                          n_entries=ZONE_ORDERS,
                          children_per_entry=TPS_PER_ORDER,
                          total_lot_per_entry=ZONE_TOTAL_LOT,
                          martingale=martingale)

    # 3) If message says "hit TP1, set BE" → arm the BE watcher
    if be_after_tp1 and tps:
        tp1 = (sorted(tps)[0] if side=="BUY" else sorted(tps, reverse=True)[0])
        LAST_BE_RULE.update({"enabled": True, "side": side, "tp1": tp1, "symbol": symbol})
        log.info(f"BE rule armed at TP1={tp1} for {symbol}")


# --------------------------
# Telegram handler
# --------------------------
SEEN = set()

def dd_exceeded(limit_pct: float) -> bool:
    if limit_pct <= 0: return False
    ai = mt5.account_info()
    if not ai or ai.balance <= 0: return False
    dd_pct = (ai.equity - ai.balance) / ai.balance * 100.0
    return dd_pct <= -abs(limit_pct)

@client.on(events.NewMessage(chats=ALLOWED_CHAT_IDS if ALLOWED_CHAT_IDS else None))
async def on_msg(event):
    if dd_exceeded(KILL_SWITCH_DD_PCT):
        log.warning("Kill-switch active (daily DD exceeded). Skipping trades.")
        return

    txt = (event.raw_text or "").replace("\u200b", "")
    fid = hashlib.sha1((txt or "").encode("utf-8")).hexdigest()
    if fid in SEEN:
        return
    SEEN.add(fid)

    first = txt.splitlines()[0] if txt else ""
    log.info(f"[IN] chat={event.chat_id} | {first!r}")

    try:
        sig = parse_message(txt)
        if not sig:
            log.info("[SKIP] not a NOW/UPDATE GOLD signal.")
            return

        connect_mt5()

        if sig["type"] == "NOW":
            handle_now(sig["side"])

        elif sig["type"] == "UPDATE":
            handle_update(sig)

        # light BE watchdog each message (safe no-op if not needed)
        symbol = pick_symbol(GOLD_SYMBOL_CANDIDATES)
        move_to_be_all(symbol, be_buffer_pts=BE_BUFFER_POINTS, magic=MAGIC)

    except Exception as e:
        log.error(f"[ERROR] {e}")

# --------------------------
# Main
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Telegram -> MT5 Bridge")
    ap.add_argument("--positions", nargs="?", const="*", metavar="SYMBOL",
                    help="Print open positions (optionally for SYMBOL, e.g., XAUUSD).")
    ap.add_argument("--positions-gold", action="store_true",
                    help="Print open GOLD positions for this bot's MAGIC.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # One-off inspection modes (no Telegram listener)
    if args.positions or args.positions_gold:
        connect_mt5()
        if args.positions_gold:
            poss = positions_for_gold(MAGIC)
            if not poss:
                log.info("No open GOLD positions for this bot's MAGIC.")
            else:
                log.info(f"GOLD positions (MAGIC={MAGIC}): {len(poss)}")
                for p in sorted(poss, key=lambda x: x.time):
                    log.info("  " + fmt_pos(p))
            sys.exit(0)

        # --positions (optional SYMBOL)
        symbol = None if args.positions in (None, "*") else args.positions
        print_positions(symbol=symbol, magic=MAGIC)
        sys.exit(0)

    if not ALLOWED_CHAT_IDS:
        log.warning("ALLOWED_CHAT_ID(S) not set. The bot will listen to ALL chats in this account.")
    log.info("Listening for GOLD NOW/UPDATE signals…")
    client.start()
    # start background BE watcher
    client.loop.create_task(be_price_watcher())
    client.run_until_disconnected()
