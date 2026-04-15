#!/usr/bin/env python3
"""Anthropic Messages API -> OpenAI Chat Completions proxy.

Accepts requests in Anthropic format, translates to OpenAI, returns
responses in Anthropic format. Designed for use with Claude Code via
ANTHROPIC_BASE_URL.

Usage:
    python proxy.py serve config.json [--debug] [--port PORT]
    ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY=dummy claude
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Union

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "debug": False,
    "compress_keep": 4,
    "compress_min": 8,
}

# Endpoint table: role -> {base_url, model, api_key}
# Roles: "opus", "sonnet", "haiku", "compress"
ENDPOINTS: dict = {}

# Fixed fallback chain: if role not configured, try next
FALLBACK_CHAIN = {
    "compress": "haiku",
    "haiku": "sonnet",
    "sonnet": "opus",
}

http_client: httpx.AsyncClient = None  # type: ignore[assignment]


def _expand_env(s: str) -> str:
    """Expand $ENV_VAR references in a string."""
    if not isinstance(s, str):
        return s
    return re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', lambda m: os.environ.get(m.group(1), m.group(0)), s)


def load_config(path: str):
    """Load JSON config file and populate ENDPOINTS, FALLBACKS, CONFIG."""
    with open(path) as f:
        cfg = json.load(f)

    # Shared defaults
    shared_api_key = _expand_env(cfg.get("api_key", ""))

    # Listen address
    listen = cfg.get("listen", "127.0.0.1:8082")
    if ":" in listen:
        host, port = listen.rsplit(":", 1)
        CONFIG["host"] = host
        CONFIG["port"] = int(port)
    else:
        CONFIG["host"] = "127.0.0.1"
        CONFIG["port"] = int(listen)

    # Compression settings
    CONFIG["compress_keep"] = cfg.get("compress_keep", 4)
    CONFIG["compress_min"] = cfg.get("compress_min", 8)
    CONFIG["debug"] = cfg.get("debug", False)

    # Endpoints
    for role, ep in cfg.get("endpoints", {}).items():
        ENDPOINTS[role] = {
            "base_url": _expand_env(ep.get("base_url", "")),
            "model": _expand_env(ep.get("model", "")),
            "api_key": _expand_env(ep.get("api_key", shared_api_key)),
            "ssl_verify": ep.get("ssl_verify", False),
        }



def resolve_endpoint(name: str) -> dict:
    """Resolve a role name or Anthropic model name to an endpoint dict.

    Fallback chain (hardcoded): compress -> haiku -> sonnet -> opus.
    """
    # Direct role lookup
    if name in ENDPOINTS:
        return ENDPOINTS[name]

    # Pattern match Anthropic model name -> role
    role = None
    name_lower = name.lower()
    if "opus" in name_lower:
        role = "opus"
    elif "haiku" in name_lower:
        role = "haiku"
    elif "sonnet" in name_lower:
        role = "sonnet"

    # Walk fallback chain: compress -> haiku -> sonnet -> opus
    current = role
    visited = set()
    while current:
        if current in ENDPOINTS:
            return ENDPOINTS[current]
        if current in visited:
            break
        visited.add(current)
        current = FALLBACK_CHAIN.get(current)

    # Last resort: first defined endpoint
    if ENDPOINTS:
        return next(iter(ENDPOINTS.values()))
    return {"base_url": "", "model": name, "api_key": "", "ssl_verify": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        verify=False,
    )
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gen_message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def to_anthropic_tool_id(openai_id: str) -> str:
    """call_abc -> toolu_abc"""
    if openai_id.startswith("call_"):
        return "toolu_" + openai_id[5:]
    return openai_id


def to_openai_tool_id(anthropic_id: str) -> str:
    """toolu_abc -> call_abc"""
    if anthropic_id.startswith("toolu_"):
        return "call_" + anthropic_id[6:]
    return anthropic_id


def _strip_chat_suffix(url: str) -> str:
    """Strip /chat/completions or /chat/completion to get the API root."""
    base = url.rstrip("/")
    for suffix in ("/chat/completions", "/chat/completion"):
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip("/")
    return base


def _chat_url(base_url: str) -> str:
    """Get the chat completions URL from a base URL."""
    return _strip_chat_suffix(base_url) + "/chat/completions"


def extract_system_text(system) -> str:
    """Extract plain text from Anthropic system param (string or block list)."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            block["text"] for block in system if block.get("type") == "text"
        )
    return str(system) if system else ""


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------


def sse_event(event_type: str, data: dict) -> str:
    data.setdefault("type", event_type)
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Debug logging (JSONL to stderr)
# ---------------------------------------------------------------------------

_REQ_COUNTER = 0


def _next_req_id() -> str:
    global _REQ_COUNTER
    _REQ_COUNTER += 1
    return f"#{_REQ_COUNTER:04d}"


def _truncate_obj(obj, depth: int = 0):
    """Deep-copy obj with large strings truncated for logging."""
    if depth > 20:
        return "...[depth limit]"
    if isinstance(obj, str):
        if len(obj) > 200 and (obj[:20].replace("+", "").replace("/", "").replace("=", "").isalnum()):
            return obj[:50] + f"...[base64, {len(obj)} chars]"
        if len(obj) > 500:
            return obj[:500] + f"...[{len(obj)} chars]"
        return obj
    if isinstance(obj, dict):
        return {k: _truncate_obj(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 30:
            return [_truncate_obj(x, depth + 1) for x in obj[:10]] + [f"...[{len(obj)} items]"]
        return [_truncate_obj(x, depth + 1) for x in obj]
    return obj


def debug_log(event: str, data=None, req_id: str = "", **extra):
    """Write one JSONL line to stderr."""
    if not CONFIG["debug"]:
        return
    record = {"ts": time.strftime("%H:%M:%S"), "event": event}
    if req_id:
        record["req"] = req_id
    record.update(extra)
    if data is not None:
        record["data"] = _truncate_obj(data)
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), file=sys.stderr)
    sys.stderr.flush()


def debug_sse(direction: str, event_str: str, req_id: str = ""):
    """Log a single SSE chunk as one JSONL line."""
    if not CONFIG["debug"]:
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
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), file=sys.stderr)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> OpenAI
# ---------------------------------------------------------------------------


def convert_content_to_openai(content):
    """Convert Anthropic content (string or block array) to OpenAI format."""
    if isinstance(content, str):
        return content

    parts = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif btype == "image":
            source = block["source"]
            if source.get("type") == "base64":
                url = f"data:{source['media_type']};base64,{source['data']}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif source.get("type") == "url":
                parts.append({"type": "image_url", "image_url": {"url": source["url"]}})

    if not parts:
        return ""
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def extract_tool_result_content(block) -> str:
    """Extract string content from a tool_result block."""
    content = block.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        text = "\n".join(texts)
    else:
        text = str(content)
    if block.get("is_error"):
        text = f"Error: {text}"
    return text


def convert_tools(anthropic_tools: list) -> list:
    """Anthropic flat tool defs -> OpenAI nested function defs."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in anthropic_tools
    ]


_TOOL_CHOICE_MAP = {"auto": "auto", "any": "required", "none": "none"}


def convert_tool_choice(anthropic_tc) -> Union[str, dict]:
    if not isinstance(anthropic_tc, dict):
        return "auto"
    tc_type = anthropic_tc.get("type", "auto")
    if tc_type == "tool":
        return {"type": "function", "function": {"name": anthropic_tc["name"]}}
    return _TOOL_CHOICE_MAP.get(tc_type, "auto")


def convert_request(body: dict, openai_model: str) -> dict:
    """Translate an Anthropic Messages API request to OpenAI Chat Completions."""
    thinking = body.get("thinking")
    has_thinking = thinking and thinking.get("type") in ("enabled", "adaptive")

    openai_messages: list[dict] = []

    system_text = extract_system_text(body.get("system"))
    if system_text:
        openai_messages.append({"role": "developer", "content": system_text})

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, list):
                tool_results = [b for b in content if b.get("type") == "tool_result"]
                other_blocks = [b for b in content if b.get("type") != "tool_result"]
            else:
                tool_results = []
                other_blocks = content

            for tr in tool_results:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": to_openai_tool_id(tr["tool_use_id"]),
                    "content": extract_tool_result_content(tr),
                })

            if other_blocks:
                converted = convert_content_to_openai(other_blocks)
                if converted:
                    openai_messages.append({"role": "user", "content": converted})

        elif role == "assistant":
            if isinstance(content, str):
                openai_messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block["text"])
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": to_openai_tool_id(block["id"]),
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })

                assistant_msg: dict = {"role": "assistant"}
                text = "\n".join(text_parts) if text_parts else None
                assistant_msg["content"] = text
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                openai_messages.append(assistant_msg)
            else:
                openai_messages.append({"role": "assistant", "content": str(content)})

    openai_body: dict = {
        "model": openai_model,
        "messages": openai_messages,
        "stream": body.get("stream", False),
    }

    max_tokens = body.get("max_tokens", 4096)
    if has_thinking:
        openai_body["max_completion_tokens"] = max_tokens
    else:
        openai_body["max_tokens"] = max_tokens

    if "temperature" in body and not has_thinking:
        openai_body["temperature"] = body["temperature"]

    if "top_p" in body:
        openai_body["top_p"] = body["top_p"]

    if "stop_sequences" in body:
        openai_body["stop"] = body["stop_sequences"]

    if "tools" in body and body["tools"]:
        openai_body["tools"] = convert_tools(body["tools"])

    if "tool_choice" in body:
        openai_body["tool_choice"] = convert_tool_choice(body["tool_choice"])

    if has_thinking:
        budget = thinking.get("budget_tokens", 0)
        if budget <= 2048:
            openai_body["reasoning_effort"] = "low"
        elif budget <= 8192:
            openai_body["reasoning_effort"] = "medium"
        else:
            openai_body["reasoning_effort"] = "high"

    if body.get("stream"):
        openai_body["stream_options"] = {"include_usage": True}

    return openai_body


# ---------------------------------------------------------------------------
# Response conversion: OpenAI -> Anthropic (non-streaming)
# ---------------------------------------------------------------------------

FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def convert_response(openai_resp: dict, anthropic_model: str) -> dict:
    """Translate an OpenAI Chat Completions response to Anthropic Messages format."""
    choice = openai_resp["choices"][0]
    msg = choice["message"]

    content_blocks: list[dict] = []

    if msg.get("reasoning_content"):
        content_blocks.append({
            "type": "thinking",
            "thinking": msg["reasoning_content"],
            "signature": "",
        })

    if msg.get("content"):
        content_blocks.append({"type": "text", "text": msg["content"]})

    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            try:
                tool_input = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                tool_input = {}
            content_blocks.append({
                "type": "tool_use",
                "id": to_anthropic_tool_id(tc["id"]),
                "name": tc["function"]["name"],
                "input": tool_input,
            })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop_reason = FINISH_REASON_MAP.get(choice.get("finish_reason"), "end_turn")
    usage = openai_resp.get("usage", {})

    return {
        "id": gen_message_id(),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": anthropic_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def error_response(status_code: int, etype: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={
        "type": "error",
        "error": {"type": etype, "message": message},
    })


def translate_openai_error(status_code: int, openai_error: Optional[dict] = None) -> JSONResponse:
    """Translate an OpenAI HTTP error into an Anthropic-format error response."""
    error_msg = "Unknown error"
    if openai_error and "error" in openai_error:
        err = openai_error["error"]
        error_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)

    if status_code == 429:
        etype = "rate_limit_error"
    elif status_code == 401:
        etype = "authentication_error"
    elif status_code >= 500:
        status_code = 529
        etype = "overloaded_error"
    else:
        status_code = 400
        etype = "invalid_request_error"

    return error_response(status_code, etype, error_msg)


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE -> Anthropic SSE
# ---------------------------------------------------------------------------


async def iter_openai_sse(response: httpx.Response) -> AsyncIterator[Union[dict, str]]:
    """Yield parsed JSON dicts (or '[DONE]') from an OpenAI SSE stream."""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                yield "[DONE]"
            else:
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


class StreamState:
    __slots__ = (
        "message_id", "anthropic_model", "block_index",
        "current_block_type", "current_tool_index",
        "input_tokens", "output_tokens",
    )

    def __init__(self, anthropic_model: str):
        self.message_id = gen_message_id()
        self.anthropic_model = anthropic_model
        self.block_index = 0
        self.current_block_type: Optional[str] = None
        self.current_tool_index: Optional[int] = None
        self.input_tokens = 0
        self.output_tokens = 0


def close_block_events(state: StreamState) -> list[str]:
    """Return SSE strings to close the current content block."""
    if state.current_block_type is None:
        return []

    events: list[str] = []
    if state.current_block_type == "thinking":
        events.append(sse_event("content_block_delta", {
            "index": state.block_index,
            "delta": {"type": "signature_delta", "signature": ""},
        }))

    events.append(sse_event("content_block_stop", {
        "index": state.block_index,
    }))
    state.block_index += 1
    state.current_block_type = None
    state.current_tool_index = None
    return events


def open_block_events(state: StreamState, block_type: str, content_block: dict) -> list[str]:
    """Close previous block if needed, then open a new one. Returns SSE strings."""
    if state.current_block_type == block_type:
        return []
    events = close_block_events(state)
    state.current_block_type = block_type
    events.append(sse_event("content_block_start", {
        "index": state.block_index,
        "content_block": content_block,
    }))
    return events


async def stream_translate(
    openai_response: httpx.Response,
    anthropic_model: str,
    req_id: str = "",
) -> AsyncIterator[str]:
    """Translate OpenAI SSE stream to Anthropic SSE events."""
    state = StreamState(anthropic_model)

    yield sse_event("message_start", {
        "message": {
            "id": state.message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": anthropic_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    yield sse_event("ping", {})

    finish_reason = None
    _chunk_count = 0

    try:
        async for chunk in iter_openai_sse(openai_response):
            if chunk == "[DONE]":
                debug_sse("in", "event: done\ndata: [DONE]\n\n", req_id=req_id)
                break

            _chunk_count += 1
            debug_sse("in", f"event: chunk\ndata: {json.dumps(chunk)}\n\n", req_id=req_id)

            choices = chunk.get("choices", [])
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {})

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                reasoning = delta.get("reasoning_content")
                if reasoning:
                    for ev in open_block_events(state, "thinking", {"type": "thinking", "thinking": ""}):
                        yield ev
                    yield sse_event("content_block_delta", {
                        "index": state.block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    })

                text = delta.get("content")
                if text:
                    for ev in open_block_events(state, "text", {"type": "text", "text": ""}):
                        yield ev
                    yield sse_event("content_block_delta", {
                        "index": state.block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })

                tc_deltas = delta.get("tool_calls")
                if tc_deltas:
                    for tc_delta in tc_deltas:
                        tc_idx = tc_delta.get("index", 0)

                        if tc_idx != state.current_tool_index:
                            for ev in close_block_events(state):
                                yield ev
                            tool_id = to_anthropic_tool_id(tc_delta.get("id", f"call_{uuid.uuid4().hex[:8]}"))
                            tool_name = tc_delta.get("function", {}).get("name", "")
                            state.current_block_type = "tool_use"
                            state.current_tool_index = tc_idx
                            yield sse_event("content_block_start", {
                                "index": state.block_index,
                                "content_block": {
                                    "type": "tool_use", "id": tool_id, "name": tool_name, "input": {},
                                },
                            })

                        args_chunk = tc_delta.get("function", {}).get("arguments", "")
                        if args_chunk:
                            yield sse_event("content_block_delta", {
                                "index": state.block_index,
                                "delta": {"type": "input_json_delta", "partial_json": args_chunk},
                            })

            usage = chunk.get("usage")
            if usage:
                state.input_tokens = usage.get("prompt_tokens", state.input_tokens)
                state.output_tokens = usage.get("completion_tokens", state.output_tokens)
    finally:
        await openai_response.aclose()

    for ev in close_block_events(state):
        yield ev

    stop_reason = FINISH_REASON_MAP.get(finish_reason, "end_turn")
    yield sse_event("message_delta", {
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": state.output_tokens},
    })

    yield sse_event("message_stop", {})

    debug_log("STREAM COMPLETE", req_id=req_id,
              chunks=_chunk_count,
              blocks=state.block_index,
              stop=stop_reason,
              input_tokens=state.input_tokens,
              output_tokens=state.output_tokens)


# ---------------------------------------------------------------------------
# OpenAI HTTP client
# ---------------------------------------------------------------------------


async def call_openai(openai_body: dict, stream: bool, base_url: str, api_key: str) -> Union[httpx.Response, dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = _chat_url(base_url)

    if stream:
        req = http_client.build_request("POST", url, json=openai_body, headers=headers)
        resp = await http_client.send(req, stream=True)
        if resp.status_code != 200:
            body = await resp.aread()
            await resp.aclose()
            raise httpx.HTTPStatusError(
                f"OpenAI returned {resp.status_code}",
                request=req,
                response=resp,
            )
        return resp
    else:
        resp = await http_client.post(url, json=openai_body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if "response" in data and "choices" not in data:
            data = data["response"]
        return data


async def _debug_stream_wrap(gen: AsyncIterator[str], req_id: str) -> AsyncIterator[str]:
    """Wrap an SSE generator to log each outgoing Anthropic event."""
    async for event_str in gen:
        debug_sse("out", event_str, req_id=req_id)
        yield event_str


# ---------------------------------------------------------------------------
# Context compression
# ---------------------------------------------------------------------------

COMPRESS_PROMPT = (
    "Summarize this conversation between a user and an AI coding assistant. "
    "Preserve: what was asked, what files were read or changed, what tools were "
    "called and their key results, decisions made, current state of the task. "
    "Be concise but complete enough that the assistant can continue the work. "
    "Output only the summary, no preamble."
)


def _has_block_type(content, btype: str) -> bool:
    if not isinstance(content, list):
        return False
    return any(b.get("type") == btype for b in content)


def find_safe_split(messages: list, keep: int) -> int:
    """Find split point that doesn't break tool_use -> tool_result chains."""
    boundary = len(messages) - keep
    if boundary < 2:
        return 0
    for i in range(boundary, 1, -1):
        msg = messages[i]
        content = msg.get("content", "")
        if msg["role"] == "user" and _has_block_type(content, "tool_result"):
            continue
        return i
    return 0


def _truncate(s: str, limit: int = 500) -> str:
    return s if len(s) <= limit else s[:limit] + f"...[{len(s)} chars]"


def serialize_messages(messages: list) -> str:
    """Serialize Anthropic messages to a readable text format for compression."""
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg.get("content", "")
        if isinstance(content, str):
            lines.append(f"{role}: {_truncate(content)}")
        elif isinstance(content, list):
            parts = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(_truncate(block.get("text", "")))
                elif btype == "tool_use":
                    inp = json.dumps(block.get("input", {}))
                    parts.append(f"[tool_use: {block.get('name', '?')}({_truncate(inp, 200)})]")
                elif btype == "tool_result":
                    sub = block.get("content", "")
                    if isinstance(sub, str):
                        text = sub
                    elif isinstance(sub, list):
                        text = " ".join(b.get("text", "") for b in sub if b.get("type") == "text")
                    else:
                        text = str(sub)
                    status = "error" if block.get("is_error") else "ok"
                    parts.append(f"[tool_result({status}): {_truncate(text, 300)}]")
                elif btype == "thinking":
                    parts.append(f"[thinking: {_truncate(block.get('thinking', ''), 200)}]")
            lines.append(f"{role}: {' '.join(parts)}")
        else:
            lines.append(f"{role}: {_truncate(str(content))}")
    return "\n".join(lines)


async def call_compress_llm(messages: list) -> str:
    """Send messages to the compression model and return summary text."""
    ep = resolve_endpoint("compress")
    serialized = serialize_messages(messages)

    compress_body = {
        "model": ep["model"],
        "messages": [
            {"role": "developer", "content": COMPRESS_PROMPT},
            {"role": "user", "content": f"<conversation>\n{serialized}\n</conversation>"},
        ],
        "max_tokens": 2048,
        "temperature": 0,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {ep['api_key']}",
        "Content-Type": "application/json",
    }
    url = _chat_url(ep["base_url"])

    resp = await http_client.post(url, json=compress_body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if "response" in data and "choices" not in data:
        data = data["response"]

    return data["choices"][0]["message"]["content"] or ""


async def compress_context(messages: list, req_id: str = "") -> list:
    """Compress old messages via cheap LLM, keeping recent ones verbatim."""
    keep = CONFIG["compress_keep"]
    min_msgs = CONFIG["compress_min"]

    if len(messages) < min_msgs:
        return messages

    split = find_safe_split(messages, keep)
    if split <= 0:
        return messages

    to_compress = messages[:split]
    tail = messages[split:]

    original_chars = sum(len(json.dumps(m.get("content", ""))) for m in messages)

    ep = resolve_endpoint("compress")
    try:
        summary = await call_compress_llm(to_compress)
    except Exception as e:
        debug_log("COMPRESS ERROR", {"error": str(e)}, req_id=req_id)
        return messages

    compressed = [
        {"role": "user", "content": f"<context-summary>\n{summary}\n</context-summary>"},
        {"role": "assistant", "content": "Understood, I have the conversation context."},
        *tail,
    ]

    debug_log("CONTEXT COMPRESSION", req_id=req_id,
              original_msgs=len(messages),
              compressed_to=len(compressed),
              kept_verbatim=len(tail),
              original_chars=original_chars,
              summary_chars=len(summary),
              model=ep["model"])

    return compressed


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def create_message(request: Request):
    req_id = _next_req_id()

    try:
        body = await request.json()
    except Exception:
        return error_response(400, "invalid_request_error", "Invalid JSON body")

    anthropic_model = body.get("model", "")
    is_stream = body.get("stream", False)
    ep = resolve_endpoint(anthropic_model)

    debug_log("ANTHROPIC REQUEST", body, req_id=req_id,
              model=anthropic_model, stream=is_stream,
              endpoint_model=ep["model"],
              messages=len(body.get("messages", [])),
              tools=len(body.get("tools", [])))

    # Context compression
    if "compress" in ENDPOINTS and "messages" in body:
        body["messages"] = await compress_context(body["messages"], req_id=req_id)

    try:
        openai_body = convert_request(body, ep["model"])
    except Exception as e:
        return error_response(400, "invalid_request_error", f"Request conversion error: {e}")

    debug_log("OPENAI REQUEST", openai_body, req_id=req_id,
              model=openai_body.get("model", ""),
              base_url=ep["base_url"],
              messages=len(openai_body.get("messages", [])))

    try:
        if is_stream:
            openai_resp = await call_openai(openai_body, stream=True,
                                            base_url=ep["base_url"], api_key=ep["api_key"])
            return StreamingResponse(
                _debug_stream_wrap(
                    stream_translate(openai_resp, anthropic_model, req_id=req_id),
                    req_id,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            openai_resp = await call_openai(openai_body, stream=False,
                                            base_url=ep["base_url"], api_key=ep["api_key"])
            debug_log("OPENAI RESPONSE", openai_resp, req_id=req_id,
                      finish=openai_resp.get("choices", [{}])[0].get("finish_reason"))
            anthropic_resp = convert_response(openai_resp, anthropic_model)
            debug_log("ANTHROPIC RESPONSE", anthropic_resp, req_id=req_id,
                      stop=anthropic_resp.get("stop_reason"))
            return JSONResponse(content=anthropic_resp)

    except httpx.HTTPStatusError as e:
        try:
            error_body = e.response.json()
        except Exception:
            error_body = None
        debug_log("OPENAI ERROR", error_body, req_id=req_id, status=e.response.status_code)
        return translate_openai_error(e.response.status_code, error_body)
    except Exception as e:
        debug_log("PROXY ERROR", {"error": str(e)}, req_id=req_id)
        return error_response(500, "api_error", str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"input_tokens": 0})

    total_chars = 0
    total_chars += len(extract_system_text(body.get("system")))

    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    total_chars += len(block.get("text", ""))
                elif btype == "tool_result":
                    sub = block.get("content", "")
                    if isinstance(sub, str):
                        total_chars += len(sub)
                    elif isinstance(sub, list):
                        for sb in sub:
                            total_chars += len(sb.get("text", ""))
                elif btype == "tool_use":
                    total_chars += len(json.dumps(block.get("input", {})))

    if "tools" in body:
        total_chars += len(json.dumps(body["tools"]))

    return JSONResponse(content={"input_tokens": max(1, total_chars // 4)})


@app.get("/v1/models")
async def list_models():
    models = [f"claude-{role}" for role in ENDPOINTS if role != "compress"] or [
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5",
    ]
    return JSONResponse(content={
        "data": [
            {"id": m, "type": "model", "display_name": m, "created_at": "2025-01-01T00:00:00Z"}
            for m in models
        ],
        "has_more": False,
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace):
    """Start the proxy server."""
    load_config(args.config)
    if args.debug:
        CONFIG["debug"] = True
    if args.port:
        CONFIG["port"] = args.port

    host = CONFIG.get("host", "127.0.0.1")
    port = CONFIG.get("port", 8082)

    print(f"Proxy starting on {host}:{port}")
    print("Endpoints:")
    for role, ep in ENDPOINTS.items():
        print(f"  {role}: {ep['base_url']} -> {ep['model']}")
    if "compress" in ENDPOINTS:
        print(f"Compression: keep={CONFIG['compress_keep']}, min={CONFIG['compress_min']}")
    if CONFIG["debug"]:
        print("Debug: ENABLED (JSONL to stderr)")
    print()
    print("Usage:")
    print(f"  ANTHROPIC_BASE_URL=http://{host}:{port} ANTHROPIC_API_KEY=dummy claude")

    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_models(args: argparse.Namespace):
    """List models available at the first endpoint."""
    load_config(args.config)
    if not ENDPOINTS:
        sys.exit("No endpoints configured")

    ep = next(iter(ENDPOINTS.values()))
    url = _strip_chat_suffix(ep["base_url"]) + "/models"
    headers = {"Authorization": f"Bearer {ep['api_key']}"}

    print(f"Fetching models from {url} ...\n")
    try:
        with httpx.Client(verify=False, timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        sys.exit(f"HTTP {e.response.status_code}: {e.response.text}")
    except Exception as e:
        sys.exit(f"Error: {e}")

    if "response" in data and "data" not in data:
        data = data["response"]

    models = data.get("data", [])
    if not models:
        print("No models returned.")
        return

    models.sort(key=lambda m: m.get("id", ""))
    print(f"Found {len(models)} models:\n")
    for m in models:
        mid = m.get("id", "?")
        owned_by = m.get("owned_by", "")
        extra = f"  (by {owned_by})" if owned_by else ""
        print(f"  {mid}{extra}")


def main():
    parser = argparse.ArgumentParser(description="Anthropic -> OpenAI proxy")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start the proxy server")
    p_serve.add_argument("config", help="Path to config.json")
    p_serve.add_argument("--debug", action="store_true", help="Enable debug logging")
    p_serve.add_argument("--port", type=int, default=0, help="Override listen port")

    p_models = sub.add_parser("models", help="List models at first configured endpoint")
    p_models.add_argument("config", help="Path to config.json")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "models":
        cmd_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
