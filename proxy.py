#!/usr/bin/env python3
"""Anthropic Messages API -> OpenAI Chat Completions proxy.

Accepts requests in Anthropic format, translates to OpenAI, returns
responses in Anthropic format. Designed for use with Claude Code via
ANTHROPIC_BASE_URL.

Usage:
    python proxy.py --port 8082 --openai-api-key sk-...
    ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY=dummy claude
"""

import argparse
import copy
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Tuple, Union

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "default_model": "gpt-5.4",
    "debug": False,
}

MODEL_MAP: dict = {
    # Will be populated at startup; any unknown model -> default_model
}

http_client: httpx.AsyncClient = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# ID helpers
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


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------


def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

_REQ_COUNTER = 0


def _next_req_id() -> str:
    global _REQ_COUNTER
    _REQ_COUNTER += 1
    return f"#{_REQ_COUNTER:04d}"


def _truncate_str(s: str, limit: int = 200) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"...[{len(s)} chars total]"


def _truncate_obj(obj, depth: int = 0):
    """Deep-copy obj with large strings truncated for readable logging."""
    if depth > 20:
        return "...[depth limit]"
    if isinstance(obj, str):
        # Base64 data — very aggressive truncation
        if len(obj) > 200 and (obj[:20].replace("+", "").replace("/", "").replace("=", "").isalnum()):
            return obj[:50] + f"...[base64, {len(obj)} chars]"
        return _truncate_str(obj, 500)
    if isinstance(obj, dict):
        return {k: _truncate_obj(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 30:
            return [_truncate_obj(x, depth + 1) for x in obj[:10]] + [f"...[{len(obj)} items total]"]
        return [_truncate_obj(x, depth + 1) for x in obj]
    return obj


def debug_log(header: str, data=None, req_id: str = "", **extra):
    """Print a formatted debug block to stderr."""
    if not CONFIG["debug"]:
        return
    ts = time.strftime("%H:%M:%S")
    tag = f"[{ts}] {req_id} " if req_id else f"[{ts}] "
    meta = " | ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    sep = "\u2550" * 50
    print(f"\n{sep}", file=sys.stderr)
    print(f"{tag}{header}" + (f" | {meta}" if meta else ""), file=sys.stderr)
    print(sep, file=sys.stderr)
    if data is not None:
        if isinstance(data, (dict, list)):
            truncated = _truncate_obj(data)
            print(json.dumps(truncated, indent=2, ensure_ascii=False), file=sys.stderr)
        else:
            print(str(data), file=sys.stderr)
    sys.stderr.flush()


def debug_sse(direction: str, event_str: str, req_id: str = ""):
    """Log a single SSE event as a compact one-liner."""
    if not CONFIG["debug"]:
        return
    ts = time.strftime("%H:%M:%S")
    # Parse the event for a compact representation
    lines = event_str.strip().split("\n")
    etype = ""
    edata = ""
    for ln in lines:
        if ln.startswith("event: "):
            etype = ln[7:]
        elif ln.startswith("data: "):
            edata = ln[6:]
    # Truncate data for one-liner
    if len(edata) > 300:
        edata = edata[:300] + "..."
    arrow = "<<" if direction == "in" else ">>"
    print(f"[{ts}] {req_id} {arrow} SSE {etype}: {edata}", file=sys.stderr)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> OpenAI
# ---------------------------------------------------------------------------


def convert_content_to_openai(content):
    """Convert Anthropic content (string or block array) to OpenAI format.

    Filters out tool_result and thinking blocks — those are handled separately.
    """
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
        # tool_result and thinking blocks are skipped here

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
    result = []
    for tool in anthropic_tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


def convert_tool_choice(anthropic_tc) -> Union[str, dict]:
    if not isinstance(anthropic_tc, dict):
        return "auto"
    tc_type = anthropic_tc.get("type", "auto")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "none":
        return "none"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": anthropic_tc["name"]}}
    return "auto"


def convert_request(body: dict) -> dict:
    """Translate an Anthropic Messages API request to OpenAI Chat Completions."""
    # Model mapping
    anthropic_model = body.get("model", "")
    openai_model = MODEL_MAP.get(anthropic_model, CONFIG["default_model"])

    # Thinking config
    thinking = body.get("thinking")
    has_thinking = thinking and thinking.get("type") in ("enabled", "adaptive")

    # Build OpenAI messages
    openai_messages: list[dict] = []

    # System prompt -> developer role message
    system = body.get("system")
    if system:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_text = "\n\n".join(
                block["text"] for block in system if block.get("type") == "text"
            )
        else:
            system_text = str(system)
        if system_text:
            openai_messages.append({"role": "developer", "content": system_text})

    # Convert messages
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            # Split tool_result blocks from the rest
            if isinstance(content, list):
                tool_results = [b for b in content if b.get("type") == "tool_result"]
                other_blocks = [b for b in content if b.get("type") != "tool_result"]
            else:
                tool_results = []
                other_blocks = content  # string

            # Emit tool messages first (they must follow the assistant's tool_calls)
            for tr in tool_results:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": to_openai_tool_id(tr["tool_use_id"]),
                    "content": extract_tool_result_content(tr),
                })

            # Emit remaining user content
            if other_blocks:
                converted = convert_content_to_openai(other_blocks)
                if converted:
                    openai_messages.append({"role": "user", "content": converted})

        elif role == "assistant":
            if isinstance(content, str):
                openai_messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                # Extract text, tool_use; drop thinking
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
                    # thinking blocks dropped

                assistant_msg: dict = {"role": "assistant"}
                text = "\n".join(text_parts) if text_parts else None
                assistant_msg["content"] = text
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                openai_messages.append(assistant_msg)
            else:
                openai_messages.append({"role": "assistant", "content": str(content)})

    # Build OpenAI request body
    openai_body: dict = {
        "model": openai_model,
        "messages": openai_messages,
        "stream": body.get("stream", False),
    }

    # max_tokens
    max_tokens = body.get("max_tokens", 4096)
    if has_thinking:
        openai_body["max_completion_tokens"] = max_tokens
    else:
        openai_body["max_tokens"] = max_tokens

    # Temperature (not allowed with reasoning models)
    if "temperature" in body and not has_thinking:
        openai_body["temperature"] = body["temperature"]

    # top_p
    if "top_p" in body:
        openai_body["top_p"] = body["top_p"]

    # Stop sequences
    if "stop_sequences" in body:
        openai_body["stop"] = body["stop_sequences"]

    # Tools
    if "tools" in body and body["tools"]:
        openai_body["tools"] = convert_tools(body["tools"])

    # Tool choice
    if "tool_choice" in body:
        openai_body["tool_choice"] = convert_tool_choice(body["tool_choice"])

    # Thinking -> reasoning_effort
    if has_thinking:
        budget = thinking.get("budget_tokens", 0)
        if budget <= 2048:
            openai_body["reasoning_effort"] = "low"
        elif budget <= 8192:
            openai_body["reasoning_effort"] = "medium"
        else:
            openai_body["reasoning_effort"] = "high"

    # Stream options for usage
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

    # Reasoning / thinking
    if msg.get("reasoning_content"):
        content_blocks.append({
            "type": "thinking",
            "thinking": msg["reasoning_content"],
            "signature": "",
        })

    # Text
    if msg.get("content"):
        content_blocks.append({"type": "text", "text": msg["content"]})

    # Tool calls
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


def translate_error(status_code: int, openai_error: Optional[dict] = None) -> Tuple[int, dict]:
    error_msg = "Unknown error"
    if openai_error and "error" in openai_error:
        error_msg = openai_error["error"].get("message", error_msg)

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

    return status_code, {"type": "error", "error": {"type": etype, "message": error_msg}}


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
    events: list[str] = []
    if state.current_block_type is None:
        return events

    if state.current_block_type == "thinking":
        events.append(sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": state.block_index,
            "delta": {"type": "signature_delta", "signature": ""},
        }))

    events.append(sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": state.block_index,
    }))
    state.block_index += 1
    state.current_block_type = None
    state.current_tool_index = None
    return events


async def stream_translate(
    openai_response: httpx.Response,
    anthropic_model: str,
    req_id: str = "",
) -> AsyncIterator[str]:
    """Translate OpenAI SSE stream to Anthropic SSE events."""
    state = StreamState(anthropic_model)

    # message_start
    yield sse_event("message_start", {
        "type": "message_start",
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

    yield sse_event("ping", {"type": "ping"})

    finish_reason = None
    _chunk_count = 0

    try:
        async for chunk in iter_openai_sse(openai_response):
            if chunk == "[DONE]":
                if CONFIG["debug"]:
                    debug_sse("in", "event: done\ndata: [DONE]\n\n", req_id=req_id)
                break

            _chunk_count += 1
            if CONFIG["debug"]:
                debug_sse("in", f"event: chunk\ndata: {json.dumps(chunk)}\n\n", req_id=req_id)

            choices = chunk.get("choices", [])
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {})

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                # --- Reasoning content ---
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    if state.current_block_type != "thinking":
                        for ev in close_block_events(state):
                            yield ev
                        state.current_block_type = "thinking"
                        yield sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": state.block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    yield sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": state.block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    })

                # --- Text content ---
                text = delta.get("content")
                if text:
                    if state.current_block_type != "text":
                        for ev in close_block_events(state):
                            yield ev
                        state.current_block_type = "text"
                        yield sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": state.block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    yield sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": state.block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })

                # --- Tool calls ---
                tc_deltas = delta.get("tool_calls")
                if tc_deltas:
                    for tc_delta in tc_deltas:
                        tc_idx = tc_delta.get("index", 0)

                        # New tool call?
                        if tc_idx != state.current_tool_index:
                            for ev in close_block_events(state):
                                yield ev

                            tool_id = to_anthropic_tool_id(tc_delta.get("id", f"call_{uuid.uuid4().hex[:8]}"))
                            tool_name = tc_delta.get("function", {}).get("name", "")

                            state.current_block_type = "tool_use"
                            state.current_tool_index = tc_idx
                            yield sse_event("content_block_start", {
                                "type": "content_block_start",
                                "index": state.block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": tool_name,
                                    "input": {},
                                },
                            })

                        # Stream arguments as input_json_delta
                        args_chunk = tc_delta.get("function", {}).get("arguments", "")
                        if args_chunk:
                            yield sse_event("content_block_delta", {
                                "type": "content_block_delta",
                                "index": state.block_index,
                                "delta": {"type": "input_json_delta", "partial_json": args_chunk},
                            })

            # Usage chunk (empty choices, has usage)
            usage = chunk.get("usage")
            if usage:
                state.input_tokens = usage.get("prompt_tokens", state.input_tokens)
                state.output_tokens = usage.get("completion_tokens", state.output_tokens)
    finally:
        await openai_response.aclose()

    # Close any open block
    for ev in close_block_events(state):
        yield ev

    # message_delta with stop reason
    stop_reason = FINISH_REASON_MAP.get(finish_reason, "end_turn")
    yield sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": state.output_tokens},
    })

    yield sse_event("message_stop", {"type": "message_stop"})

    debug_log("STREAM COMPLETE", req_id=req_id,
              chunks=_chunk_count,
              blocks=state.block_index,
              stop=stop_reason,
              input_tokens=state.input_tokens,
              output_tokens=state.output_tokens)


# ---------------------------------------------------------------------------
# OpenAI HTTP client
# ---------------------------------------------------------------------------


async def call_openai(openai_body: dict, stream: bool) -> Union[httpx.Response, dict]:
    headers = {
        "Authorization": f"Bearer {CONFIG['openai_api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{CONFIG['openai_base_url'].rstrip('/')}/chat/completions"

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
        return resp.json()


async def _debug_stream_wrap(gen: AsyncIterator[str], req_id: str) -> AsyncIterator[str]:
    """Wrap an SSE generator to log each outgoing Anthropic event."""
    async for event_str in gen:
        debug_sse("out", event_str, req_id=req_id)
        yield event_str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def create_message(request: Request):
    req_id = _next_req_id()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "Invalid JSON body"},
        })

    anthropic_model = body.get("model", "")
    is_stream = body.get("stream", False)

    debug_log("ANTHROPIC REQUEST", body, req_id=req_id,
              model=anthropic_model, stream=is_stream,
              messages=len(body.get("messages", [])),
              tools=len(body.get("tools", [])))

    try:
        openai_body = convert_request(body)
    except Exception as e:
        return JSONResponse(status_code=400, content={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": f"Request conversion error: {e}"},
        })

    debug_log("OPENAI REQUEST", openai_body, req_id=req_id,
              model=openai_body.get("model", ""),
              messages=len(openai_body.get("messages", [])))

    try:
        if is_stream:
            openai_resp = await call_openai(openai_body, stream=True)
            return StreamingResponse(
                _debug_stream_wrap(
                    stream_translate(openai_resp, anthropic_model, req_id=req_id),
                    req_id,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            openai_resp = await call_openai(openai_body, stream=False)
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
        status, err = translate_error(e.response.status_code, error_body)
        return JSONResponse(status_code=status, content=err)
    except Exception as e:
        debug_log("PROXY ERROR", {"error": str(e)}, req_id=req_id)
        return JSONResponse(status_code=500, content={
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        })


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"input_tokens": 0})

    total_chars = 0

    # System prompt
    system = body.get("system", "")
    if isinstance(system, str):
        total_chars += len(system)
    elif isinstance(system, list):
        for block in system:
            total_chars += len(block.get("text", ""))

    # Messages
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

    # Tool definitions
    if "tools" in body:
        total_chars += len(json.dumps(body["tools"]))

    estimated_tokens = max(1, total_chars // 4)

    return JSONResponse(content={"input_tokens": estimated_tokens})


@app.get("/v1/models")
async def list_models():
    models = list(MODEL_MAP.keys()) or [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ]
    return JSONResponse(content={
        "data": [
            {
                "id": m,
                "type": "model",
                "display_name": m,
                "created_at": "2025-01-01T00:00:00Z",
            }
            for m in models
        ],
        "has_more": False,
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Anthropic -> OpenAI proxy")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PROXY_PORT", "8082")))
    parser.add_argument("--host", default=os.environ.get("PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--default-model", default=os.environ.get("DEFAULT_MODEL", "gpt-5.4"))
    parser.add_argument(
        "--model-map",
        nargs="*",
        metavar="CLAUDE=OPENAI",
        help="Model mapping pairs, e.g. claude-sonnet-4-6=gpt-5.4",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"),
        help="Enable debug logging of all requests/responses to stderr",
    )
    args = parser.parse_args()

    if not args.openai_api_key:
        parser.error("--openai-api-key or OPENAI_API_KEY is required")

    CONFIG["openai_api_key"] = args.openai_api_key
    CONFIG["openai_base_url"] = args.openai_base_url
    CONFIG["default_model"] = args.default_model
    CONFIG["debug"] = args.debug

    # Default model map
    MODEL_MAP.update({
        "claude-opus-4-6": args.default_model,
        "claude-sonnet-4-6": args.default_model,
        "claude-sonnet-4-5-20250514": args.default_model,
        "claude-sonnet-4-20250514": args.default_model,
        "claude-3-5-sonnet-20241022": args.default_model,
        "claude-3-5-haiku-20241022": args.default_model,
        "claude-haiku-4-5": args.default_model,
    })

    # Custom model mappings from CLI
    if args.model_map:
        for pair in args.model_map:
            if "=" in pair:
                k, v = pair.split("=", 1)
                MODEL_MAP[k.strip()] = v.strip()

    print(f"Proxy starting on {args.host}:{args.port}")
    print(f"OpenAI base URL: {args.openai_base_url}")
    print(f"Default model: {args.default_model}")
    print(f"Model map: {MODEL_MAP}")
    if CONFIG["debug"]:
        print("Debug logging: ENABLED (stderr)")
    print()
    print("Usage:")
    print(f"  ANTHROPIC_BASE_URL=http://{args.host}:{args.port} ANTHROPIC_API_KEY=dummy claude")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
