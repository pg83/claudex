import json
import uuid
import hashlib

from typing import AsyncIterator, Optional

from starlette.responses import JSONResponse

import claudex.log as lg


# ---------------------------------------------------------------------------
# Proxy-handled tools / forwarded headers
# ---------------------------------------------------------------------------

PROXY_HANDLED_TOOLS = {"WebFetch", "WebSearch", "web_search"}

FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

FORWARD_HEADERS = ("anthropic-beta", "anthropic-version", "user-agent")


# ---------------------------------------------------------------------------
# Tool ID conversions
# ---------------------------------------------------------------------------


def to_anthropic_tool_id(openai_id: str) -> str:
    if openai_id.startswith("call_"):
        return "toolu_" + openai_id[5:]

    return openai_id


def to_openai_tool_id(anthropic_id: str) -> str:
    if anthropic_id.startswith("toolu_"):
        return "call_" + anthropic_id[6:]

    return anthropic_id


# ---------------------------------------------------------------------------
# Message content helpers
# ---------------------------------------------------------------------------


def extract_system_text(system) -> str:
    if isinstance(system, str):
        return system

    if isinstance(system, list):
        return "\n\n".join(
            block["text"] for block in system if block.get("type") == "text"
        )

    return str(system) if system else ""


def extract_text_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]

        return " ".join(texts)

    return ""


def extract_msg_text(msg: dict) -> str:
    return extract_text_content(msg.get("content", ""))


def extract_last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return extract_text_content(msg.get("content", ""))

    return ""


def session_id(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user":
            text = extract_text_content(m.get("content", ""))

            if text:
                return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    return "nosess"


def count_content_chars(content) -> int:
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


def extract_proxy_tool_uses(anthropic_resp: dict) -> list:
    return [
        b for b in anthropic_resp.get("content", [])
        if b.get("type") == "tool_use" and b.get("name") in PROXY_HANDLED_TOOLS
    ]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def gen_message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def sse_event(event_type: str, data: dict) -> str:
    data.setdefault("type", event_type)

    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def debug_stream_wrap(gen: AsyncIterator[str], config: dict, req_id: str) -> AsyncIterator[str]:
    async for event_str in gen:
        lg.debug_sse(config, "out", event_str, req_id=req_id)

        yield event_str


# ---------------------------------------------------------------------------
# Error responses
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


