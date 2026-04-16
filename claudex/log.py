"""Server-side logging helpers.

- log(msg, req_id): info line to stderr (always on)
- debug_log(config, event, ...): JSONL to stdout when config["debug"]
- debug_sse(config, direction, event_str, req_id): SSE chunk to stdout
"""

import sys
import json
import time
import random


_REQ_COUNTER = 0
_SID_COLORS: dict[str, str] = {}
_MAX_SID_COLORS = 1_000_000
_RESET = "\033[0m"


def next_req_id() -> str:
    global _REQ_COUNTER

    _REQ_COUNTER += 1

    return f"#{_REQ_COUNTER:04d}"


def _color_for(sid: str) -> str:
    c = _SID_COLORS.get(sid)

    if c is not None:
        return c

    if len(_SID_COLORS) >= _MAX_SID_COLORS:
        _SID_COLORS.clear()

    r = random.randint(120, 255)
    g = random.randint(120, 255)
    b = random.randint(120, 255)
    c = f"\033[38;2;{r};{g};{b}m"
    _SID_COLORS[sid] = c

    return c


def log(msg: str, sid: str = ""):
    ts = time.strftime("%H:%M:%S")

    if sid:
        color = _color_for(sid)
        print(f"{ts} | {color}{sid}{_RESET} | {msg}", file=sys.stderr, flush=True)
    else:
        print(f"{ts} | {msg}", file=sys.stderr, flush=True)


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"

    if n < 1024 * 1024:
        return f"{n/1024:.1f}KB"

    return f"{n/1024/1024:.1f}MB"


def debug_log(config: dict, event: str, data=None, req_id: str = "", sid: str = "", **extra):
    if not config.get("debug"):
        return

    record = {"ts": time.strftime("%H:%M:%S"), "event": event}

    if req_id:
        record["req"] = req_id

    if sid:
        record["sid"] = sid

    record.update(extra)

    if data is not None:
        record["data"] = data

    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)


def debug_sse(config: dict, direction: str, event_str: str, req_id: str = "", sid: str = ""):
    if not config.get("debug"):
        return

    lines = event_str.strip().split("\n")
    etype = ""
    edata = ""

    for ln in lines:
        if ln.startswith("event: "):
            etype = ln[7:]
        elif ln.startswith("data: "):
            edata = ln[6:]

    if len(edata) > 300:
        edata = edata[:300] + "..."

    record = {
        "ts": time.strftime("%H:%M:%S"),
        "event": "sse",
        "dir": direction,
    }

    if req_id:
        record["req"] = req_id

    if sid:
        record["sid"] = sid

    if etype:
        record["sse_type"] = etype

    if edata:
        record["sse_data"] = edata

    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)
