"""serve subcommand: Starlette proxy server."""

import os
import sys
import json
import time
import uuid
import httpx
import uvicorn
import argparse

from contextlib import asynccontextmanager

from typing import AsyncIterator, Optional, Union

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

import claudex.common as cx
import claudex.log as lg
import claudex.rag as rag_mod


http_client: httpx.AsyncClient = None  # type: ignore[assignment]
rag_instance: rag_mod.RAG = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _chat_url(base_url: str) -> str:
    return cx._strip_chat_suffix(base_url) + "/chat/completions"


def _messages_url(base_url: str) -> str:
    base = base_url.rstrip("/")

    if base.endswith("/messages"):
        return base

    return base + "/messages"


# ---------------------------------------------------------------------------
# Starlette app with shared http client
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Starlette):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        verify=False,
    )
    yield
    await http_client.aclose()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def gen_message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def to_anthropic_tool_id(openai_id: str) -> str:
    if openai_id.startswith("call_"):
        return "toolu_" + openai_id[5:]
    return openai_id


def to_openai_tool_id(anthropic_id: str) -> str:
    if anthropic_id.startswith("toolu_"):
        return "call_" + anthropic_id[6:]
    return anthropic_id


def extract_system_text(system) -> str:
    if isinstance(system, str):
        return system

    if isinstance(system, list):
        return "\n\n".join(
            block["text"] for block in system if block.get("type") == "text"
        )

    return str(system) if system else ""


def sse_event(event_type: str, data: dict) -> str:
    data.setdefault("type", event_type)
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Message/content helpers
# ---------------------------------------------------------------------------


def _extract_text_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        return " ".join(texts)

    return ""


def _extract_msg_text(msg: dict) -> str:
    return _extract_text_content(msg.get("content", ""))


def _extract_last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _extract_text_content(msg.get("content", ""))

    return ""


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI request conversion
# ---------------------------------------------------------------------------


def convert_content_to_openai(content):
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


def _convert_user_msg(content) -> list[dict]:
    if isinstance(content, list):
        tool_results = [b for b in content if b.get("type") == "tool_result"]
        other_blocks = [b for b in content if b.get("type") != "tool_result"]
    else:
        tool_results = []
        other_blocks = content

    result = []

    for tr in tool_results:
        result.append({
            "role": "tool",
            "tool_call_id": to_openai_tool_id(tr["tool_use_id"]),
            "content": extract_tool_result_content(tr),
        })

    if other_blocks:
        converted = convert_content_to_openai(other_blocks)

        if converted:
            result.append({"role": "user", "content": converted})

    return result


def _convert_assistant_msg(content) -> dict:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    if isinstance(content, list):
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

        return assistant_msg

    return {"role": "assistant", "content": str(content)}


def convert_request(body: dict, openai_model: str) -> dict:
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
            openai_messages.extend(_convert_user_msg(content))
        elif role == "assistant":
            openai_messages.append(_convert_assistant_msg(content))

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
# OpenAI -> Anthropic response conversion (non-streaming)
# ---------------------------------------------------------------------------

FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def convert_response(openai_resp: dict, anthropic_model: str) -> dict:
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
    config: dict,
    req_id: str = "",
) -> AsyncIterator[str]:
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
    _raw_chunks = []

    try:
        async for chunk in iter_openai_sse(openai_response):
            if chunk == "[DONE]":
                lg.debug_sse(config, "in", "event: done\ndata: [DONE]\n\n", req_id=req_id)
                break

            _chunk_count += 1
            _raw_chunks.append(chunk)
            lg.debug_sse(config, "in", f"event: chunk\ndata: {json.dumps(chunk)}\n\n", req_id=req_id)

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

    lg.log(f"done | {state.input_tokens}in/{state.output_tokens}out | {stop_reason}",
        req_id=req_id)
    lg.debug_log(config, "OPENAI RESPONSE (stream)", _raw_chunks, req_id=req_id,
              chunks=_chunk_count,
              stop=stop_reason,
              input_tokens=state.input_tokens,
              output_tokens=state.output_tokens)


# ---------------------------------------------------------------------------
# OpenAI HTTP client
# ---------------------------------------------------------------------------


async def call_openai(openai_body: dict, stream: bool, base_url: str, api_key: str, client_headers=None) -> Union[httpx.Response, dict]:
    headers = {"Content-Type": "application/json"}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif client_headers:
        auth = client_headers.get("authorization")

        if auth:
            headers["Authorization"] = auth

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


async def fake_stream_from_response(anthropic_resp: dict, req_id: str = "") -> AsyncIterator[str]:
    yield sse_event("message_start", {"message": {
        "id": anthropic_resp.get("id", gen_message_id()),
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": anthropic_resp.get("model", ""),
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }})
    yield sse_event("ping", {})

    for idx, block in enumerate(anthropic_resp.get("content", [])):
        btype = block.get("type", "text")

        if btype == "text":
            yield sse_event("content_block_start", {
                "index": idx, "content_block": {"type": "text", "text": ""},
            })
            yield sse_event("content_block_delta", {
                "index": idx, "delta": {"type": "text_delta", "text": block.get("text", "")},
            })
            yield sse_event("content_block_stop", {"index": idx})
        elif btype == "thinking":
            yield sse_event("content_block_start", {
                "index": idx, "content_block": {"type": "thinking", "thinking": ""},
            })
            yield sse_event("content_block_delta", {
                "index": idx, "delta": {"type": "thinking_delta", "thinking": block.get("thinking", "")},
            })
            yield sse_event("content_block_delta", {
                "index": idx, "delta": {"type": "signature_delta", "signature": ""},
            })
            yield sse_event("content_block_stop", {"index": idx})
        elif btype == "tool_use":
            yield sse_event("content_block_start", {
                "index": idx, "content_block": {
                    "type": "tool_use", "id": block.get("id", ""), "name": block.get("name", ""), "input": {},
                },
            })
            yield sse_event("content_block_delta", {
                "index": idx, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))},
            })
            yield sse_event("content_block_stop", {"index": idx})

    yield sse_event("message_delta", {
        "delta": {"stop_reason": anthropic_resp.get("stop_reason", "end_turn"), "stop_sequence": None},
        "usage": {"output_tokens": anthropic_resp.get("usage", {}).get("output_tokens", 0)},
    })
    yield sse_event("message_stop", {})

    usage = anthropic_resp.get("usage", {})
    lg.log(f"done | {usage.get('input_tokens',0)}in/{usage.get('output_tokens',0)}out | {anthropic_resp.get('stop_reason','')}",
        req_id=req_id)


async def _debug_stream_wrap(gen: AsyncIterator[str], config: dict, req_id: str) -> AsyncIterator[str]:
    async for event_str in gen:
        lg.debug_sse(config, "out", event_str, req_id=req_id)
        yield event_str


# ---------------------------------------------------------------------------
# Proxy-handled tools
# ---------------------------------------------------------------------------

PROXY_HANDLED_TOOLS = {"WebFetch", "WebSearch", "web_search"}


_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


async def execute_proxy_tool(config: dict, name: str, tool_input: dict, req_id: str = "") -> str:
    if name == "WebFetch":
        url = tool_input.get("url", "")
        lg.log(f"web_fetch {url}", req_id=req_id)
        resp = await http_client.get(url, headers=_FETCH_HEADERS, timeout=15, follow_redirects=True)
        result = resp.text
        lg.debug_log(config, "TOOL EXECUTE", {"name": name, "input": tool_input, "output": result}, req_id=req_id)
        return result

    if name in ("WebSearch", "web_search"):
        query = tool_input.get("query", "")
        lg.log(f"web_search {query}", req_id=req_id)
        resp = await http_client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=_FETCH_HEADERS,
            timeout=15,
            follow_redirects=True,
        )
        result = resp.text
        lg.debug_log(config, "TOOL EXECUTE", {"name": name, "input": tool_input, "output": result}, req_id=req_id)
        return result

    return f"Unknown tool: {name}"


def _extract_proxy_tool_calls(openai_resp: dict) -> list:
    choices = openai_resp.get("choices", [])

    if not choices:
        return []

    msg = choices[0].get("message", {})
    tool_calls = msg.get("tool_calls") or []
    return [tc for tc in tool_calls if tc.get("function", {}).get("name") in PROXY_HANDLED_TOOLS]


# ---------------------------------------------------------------------------
# Context compression
# ---------------------------------------------------------------------------

COMPRESS_PROMPT = (
    "You are a context compressor. You received a JSON array of conversation messages. "
    "Your ONLY job is to retell what happened in the conversation turn by turn. "
    "Do NOT analyze, critique, or suggest improvements to any code you see. "
    "Do NOT write code. Do NOT offer opinions.\n\n"
    "For each turn, write:\n"
    "- What the user asked or said\n"
    "- What the assistant did (which files it read/wrote, what tools it called, what it responded)\n"
    "- Key results, errors, decisions\n\n"
    "Preserve file paths, function names, specific values, and code snippets that are important for context. "
    "A new assistant will use your summary to continue the conversation, so do not lose any context. "
    "Output only the summary, no preamble."
)


def serialize_messages(messages: list) -> str:
    return json.dumps(messages, ensure_ascii=False)


def _build_compress_body(messages: list, model: str) -> dict:
    serialized = serialize_messages(messages)
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": f"<conversation>\n{serialized}\n</conversation>"},
            {"role": "assistant", "content": "I've read the conversation log. What should I do with it?"},
            {"role": "user", "content": COMPRESS_PROMPT},
        ],
        "max_tokens": max(4096, len(serialized) // 4),
        "temperature": 0,
        "stream": False,
        "full": messages,
    }


async def call_compress_llm(config: dict, messages: list, req_id: str = "") -> str:
    ep = cx.resolve_endpoint(config, "compress")
    compress_body = _build_compress_body(messages, ep["model"])
    lg.debug_log(config, "COMPRESS_REQ", compress_body, req_id=req_id)

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

    lg.debug_log(config, "COMPRESS_RESP", data, req_id=req_id)

    return data["choices"][0]["message"]["content"] or ""


def _collapse_messages(messages: list) -> list:
    result = []
    user_text = []

    def flush_user_text():
        if user_text:
            result.append({"role": "user", "content": "\n\n".join(user_text)})
            user_text.clear()

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                if content.strip():
                    user_text.append(content)
            elif isinstance(content, list):
                has_tool_result = any(b.get("type") == "tool_result" for b in content)

                if has_tool_result:
                    flush_user_text()
                    result.append(msg)
                else:
                    for block in content:
                        if block.get("type") == "text" and block.get("text", "").strip():
                            user_text.append(block["text"])

        elif role == "assistant":
            flush_user_text()

            if isinstance(content, str):
                if content.strip():
                    result.append(msg)
            elif isinstance(content, list):
                blocks = [b for b in content if b.get("type") != "thinking"]

                if blocks:
                    result.append({"role": "assistant", "content": blocks})

    flush_user_text()
    return result


async def compress_context(config: dict, messages: list, req_id: str = "") -> list:
    keep = config["compress_keep"]
    min_msgs = config["compress_min"]

    if len(messages) < min_msgs:
        return messages

    split = len(messages) - keep

    if split < 2:
        return messages

    to_collapse = messages[:split]
    tail = messages[split:]

    collapsed = _collapse_messages(to_collapse)
    compressed = [*collapsed, *tail]

    original_total = len(json.dumps(messages, ensure_ascii=False))
    compressed_bytes = len(json.dumps(compressed, ensure_ascii=False))
    ratio = original_total / compressed_bytes if compressed_bytes else 0
    lg.log(f"compress {len(messages)}→{len(compressed)} msgs, {lg.human_bytes(original_total)}→{lg.human_bytes(compressed_bytes)}, {ratio:.1f}x",
        req_id=req_id)
    lg.debug_log(config, "CONTEXT COMPRESSION", compressed, req_id=req_id,
              original_msgs=len(messages),
              compressed_to=len(compressed),
              kept_verbatim=len(tail),
              original_total=original_total,
              compressed_bytes=compressed_bytes)

    return compressed


# ---------------------------------------------------------------------------
# Proxy tool loop (OpenAI path)
# ---------------------------------------------------------------------------


async def _proxy_tool_loop(openai_body: dict, ep: dict, config: dict, req_id: str, client_headers=None) -> dict:
    for _tool_iter in range(6):
        lg.debug_log(config, "OPENAI REQUEST", openai_body, req_id=req_id,
                  model=openai_body.get("model", ""),
                  base_url=ep["base_url"],
                  messages=len(openai_body.get("messages", [])))

        openai_resp = await call_openai(openai_body, stream=False,
                                        base_url=ep["base_url"], api_key=ep["api_key"],
                                        client_headers=client_headers)
        lg.debug_log(config, "OPENAI RESPONSE", openai_resp, req_id=req_id,
                  finish=openai_resp.get("choices", [{}])[0].get("finish_reason"))

        proxy_tools = _extract_proxy_tool_calls(openai_resp)

        if not proxy_tools:
            break

        assistant_msg = openai_resp["choices"][0]["message"]
        openai_body["messages"].append(assistant_msg)

        for tc in proxy_tools:
            fn = tc["function"]
            tool_name = fn["name"]

            try:
                tool_input = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_input = {}

            try:
                result = await execute_proxy_tool(config, tool_name, tool_input, req_id=req_id)
            except Exception as e:
                result = f"Error: {e}"
                lg.log(f"tool error {tool_name}: {e}", req_id=req_id)

            openai_body["messages"].append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    return openai_resp


# ---------------------------------------------------------------------------
# Anthropic passthrough upstream
# ---------------------------------------------------------------------------

_FORWARD_HEADERS = ("anthropic-beta", "anthropic-version", "user-agent")


async def call_anthropic(body: dict, base_url: str, api_key: str, client_headers=None) -> dict:
    headers = {"content-type": "application/json"}

    if client_headers:
        for h in _FORWARD_HEADERS:
            v = client_headers.get(h)

            if v:
                headers[h] = v

    if api_key:
        headers["x-api-key"] = api_key
    elif client_headers:
        auth = client_headers.get("authorization")

        if auth:
            headers["authorization"] = auth

        xapi = client_headers.get("x-api-key")

        if xapi:
            headers["x-api-key"] = xapi

    headers.setdefault("anthropic-version", "2023-06-01")

    url = _messages_url(base_url)
    resp = await http_client.post(url, json=body, headers=headers)
    resp.raise_for_status()
    return resp.json()


def _extract_proxy_tool_uses(anthropic_resp: dict) -> list:
    return [
        b for b in anthropic_resp.get("content", [])
        if b.get("type") == "tool_use" and b.get("name") in PROXY_HANDLED_TOOLS
    ]


async def _proxy_tool_loop_anthropic(body: dict, ep: dict, config: dict, req_id: str, client_headers=None) -> dict:
    resp: dict = {}

    for _ in range(6):
        lg.debug_log(config, "ANTHROPIC UPSTREAM REQUEST", body, req_id=req_id,
                  model=body.get("model", ""),
                  base_url=ep["base_url"],
                  messages=len(body.get("messages", [])))
        resp = await call_anthropic(body, base_url=ep["base_url"], api_key=ep["api_key"],
                                    client_headers=client_headers)
        lg.debug_log(config, "ANTHROPIC UPSTREAM RESPONSE", resp, req_id=req_id,
                  stop=resp.get("stop_reason"))

        proxy_uses = _extract_proxy_tool_uses(resp)

        if not proxy_uses:
            break

        body["messages"].append({"role": "assistant", "content": resp.get("content", [])})

        tool_results = []

        for tu in proxy_uses:
            try:
                result = await execute_proxy_tool(config, tu["name"], tu.get("input", {}), req_id=req_id)
            except Exception as e:
                result = f"Error: {e}"
                lg.log(f"tool error {tu['name']}: {e}", req_id=req_id)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result,
            })

        body["messages"].append({"role": "user", "content": tool_results})

    return resp


# ---------------------------------------------------------------------------
# RAG enrichment
# ---------------------------------------------------------------------------


def _enrich_with_rag(body: dict, config: dict, req_id: str):
    chunk_size = config.get("rag_chunk_size", 2000)

    for msg in body.get("messages", []):
        text = _extract_msg_text(msg)

        if text:
            rag_instance.add(f"conversation/{req_id}/{msg['role']}", text, chunk_size)

    last_text = _extract_last_user_text(body.get("messages", []))

    if not last_text:
        return

    rag_results = rag_instance.search(last_text, config.get("rag_max_results", 3))

    if not rag_results:
        return

    rag_block = "\n".join(
        f"File: {r['path']} (chunk {r['idx']})\n---\n{r['content']}\n---"
        for r in rag_results
    )
    messages = body.get("messages", [])

    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            suffix = f"\n---\n<rag>\n{rag_block}\n</rag>"

            if isinstance(content, str):
                msg["content"] = content + suffix
            elif isinstance(content, list):
                content.append({"type": "text", "text": suffix})

            break

    hits = " | ".join(f"{r['path']}:{r['idx']}({r['rank']:.1f})" for r in rag_results)
    lg.log(f"rag: {len(rag_results)} chunks — {hits}", req_id=req_id)
    lg.debug_log(config, "RAG", {"query": last_text, "results": rag_results}, req_id=req_id)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


async def create_message(request: Request):
    config = request.app.state.config
    req_id = lg.next_req_id()

    try:
        body = await request.json()
    except Exception:
        return error_response(400, "invalid_request_error", "Invalid JSON body")

    anthropic_model = body.get("model", "")
    is_stream = body.get("stream", False)
    ep = cx.resolve_endpoint(config, anthropic_model)
    n_msgs = len(body.get("messages", []))
    n_tools = len(body.get("tools", []))
    body_bytes = len(json.dumps(body))
    stream_tag = "stream" if is_stream else "sync"

    lg.log(f"{anthropic_model} -> {ep['model']} | {n_msgs} msgs, {n_tools} tools, ~{body_bytes//4}tok, {lg.human_bytes(body_bytes)}, {stream_tag}",
        req_id=req_id)

    lg.debug_log(config, "ANTHROPIC REQUEST", body, req_id=req_id,
              model=anthropic_model, stream=is_stream,
              endpoint_model=ep["model"],
              messages=n_msgs, tools=n_tools)

    if "compress" in config["endpoints"] and "messages" in body:
        body["messages"] = await compress_context(config, body["messages"], req_id=req_id)

    if rag_instance is not None:
        _enrich_with_rag(body, config, req_id)

    try:
        if ep.get("protocol") == "anthropic":
            upstream_body = dict(body)
            upstream_body["messages"] = list(body.get("messages", []))
            upstream_body["model"] = ep["model"]
            upstream_body["stream"] = False
            anthropic_resp = await _proxy_tool_loop_anthropic(
                upstream_body, ep, config, req_id, client_headers=request.headers,
            )
            anthropic_resp["model"] = anthropic_model
        else:
            openai_body = convert_request(body, ep["model"])
            openai_body["stream"] = False
            openai_body.pop("stream_options", None)
            openai_resp = await _proxy_tool_loop(openai_body, ep, config, req_id, client_headers=request.headers)
            anthropic_resp = convert_response(openai_resp, anthropic_model)

        if is_stream:
            lg.debug_log(config, "ANTHROPIC RESPONSE", anthropic_resp, req_id=req_id,
                      stop=anthropic_resp.get("stop_reason"))
            return StreamingResponse(
                _debug_stream_wrap(
                    fake_stream_from_response(anthropic_resp, req_id=req_id),
                    config,
                    req_id,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        else:
            usage = anthropic_resp.get("usage", {})
            lg.log(f"done | {usage.get('input_tokens',0)}in/{usage.get('output_tokens',0)}out | {anthropic_resp.get('stop_reason','')}",
                req_id=req_id)
            lg.debug_log(config, "ANTHROPIC RESPONSE", anthropic_resp, req_id=req_id,
                      stop=anthropic_resp.get("stop_reason"))
            return JSONResponse(content=anthropic_resp)

    except httpx.HTTPStatusError as e:
        try:
            error_body = e.response.json()
        except Exception:
            error_body = None

        lg.log(f"ERROR {e.response.status_code}", req_id=req_id)
        lg.debug_log(config, "OPENAI ERROR", error_body, req_id=req_id, status=e.response.status_code)
        return translate_openai_error(e.response.status_code, error_body)

    except Exception as e:
        lg.log(f"ERROR {e}", req_id=req_id)
        lg.debug_log(config, "PROXY ERROR", {"error": str(e)}, req_id=req_id)
        return error_response(500, "api_error", str(e))


def _count_content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)

    if isinstance(content, list):
        total = 0

        for block in content:
            btype = block.get("type", "")

            if btype == "text":
                total += len(block.get("text", ""))
            elif btype == "tool_result":
                sub = block.get("content", "")

                if isinstance(sub, str):
                    total += len(sub)
                elif isinstance(sub, list):
                    for sb in sub:
                        total += len(sb.get("text", ""))

            elif btype == "tool_use":
                total += len(json.dumps(block.get("input", {})))

        return total

    return 0


async def count_tokens(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"input_tokens": 0})

    total_chars = len(extract_system_text(body.get("system")))

    for msg in body.get("messages", []):
        total_chars += _count_content_chars(msg.get("content", ""))

    if "tools" in body:
        total_chars += len(json.dumps(body["tools"]))

    return JSONResponse(content={"input_tokens": max(1, total_chars // 4)})


async def list_models(request: Request):
    config = request.app.state.config
    endpoints = config.get("endpoints", {})
    models = [f"claude-{role}" for role in endpoints if role != "compress"] or [
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5",
    ]
    return JSONResponse(content={
        "data": [
            {"id": m, "type": "model", "display_name": m, "created_at": "2025-01-01T00:00:00Z"}
            for m in models
        ],
        "has_more": False,
    })


app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/v1/messages", create_message, methods=["POST"]),
        Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
        Route("/v1/models", list_models, methods=["GET"]),
    ],
)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace):
    global rag_instance

    config = cx.load_config(args.config)

    if args.debug:
        config["debug"] = True

    if args.port:
        config["port"] = args.port

    app.state.config = config

    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8082)
    endpoints = config.get("endpoints", {})

    def info(msg): print(msg, file=sys.stderr, flush=True)

    info(f"Proxy starting on {host}:{port}")
    info("Endpoints:")

    for role, ep in endpoints.items():
        info(f"  {role} [{ep['protocol']}]: {ep['base_url']} -> {ep['model']}")

    if "compress" in endpoints:
        info(f"Compression: keep={config['compress_keep']}, min={config['compress_min']}")

    rag_dirs = config.get("rag_dirs", [])

    if rag_dirs:
        exts = config.get("rag_extensions")
        chunk_size = config.get("rag_chunk_size", 2000)
        dirs = [os.path.expanduser(d) for d in rag_dirs]
        rag_instance = rag_mod.RAG(dirs, set(exts) if exts else None, chunk_size)
        info(f"RAG: {rag_instance.n_files} files, {rag_instance.n_chunks} chunks from {', '.join(dirs)}")

    if config["debug"]:
        info("Debug: ENABLED (JSONL to stdout, redirect with > debug.jsonl)")

    info("")
    info("Usage:")
    info(f"  ANTHROPIC_BASE_URL=http://{host}:{port} ANTHROPIC_API_KEY=dummy claude")

    uvicorn.run(app, host=host, port=port, log_level="info")
