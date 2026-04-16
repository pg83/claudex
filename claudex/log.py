"""Server-side logging helpers.

- log(msg, req_id): info line to stderr (always on)
- debug_log(config, event, ...): JSONL to stdout when config["debug"]
- debug_sse(config, direction, event_str, req_id): SSE chunk to stdout
"""

import sys
import time
import json


_REQ_COUNTER = 0


def next_req_id() -> str:
    global _REQ_COUNTER

    _REQ_COUNTER += 1

    return f"#{_REQ_COUNTER:04d}"


def log(msg: str, req_id: str = ""):
    ts = time.strftime("%H:%M:%S")
    prefix = f"{ts} {req_id} " if req_id else f"{ts} "

    print(f"{prefix}{msg}", file=sys.stderr, flush=True)


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"

    if n < 1024 * 1024:
        return f"{n/1024:.1f}KB"

    return f"{n/1024/1024:.1f}MB"


def debug_log(config: dict, event: str, data=None, req_id: str = "", **extra):
    if not config.get("debug"):
        return

    record = {"ts": time.strftime("%H:%M:%S"), "event": event}

    if req_id:
        record["req"] = req_id

    record.update(extra)

    if data is not None:
        record["data"] = data

    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)


def debug_sse(config: dict, direction: str, event_str: str, req_id: str = ""):
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
        "req": req_id,
        "dir": direction,
    }

    if etype:
        record["sse_type"] = etype

    if edata:
        record["sse_data"] = edata

    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)
