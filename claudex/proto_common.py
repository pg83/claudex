import json
import uuid

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


# ---------------------------------------------------------------------------
# Context compression helpers (data shaping only; call lives in ProxyServer)
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


def build_compress_body(messages: list, model: str) -> dict:
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


def collapse_messages(messages: list) -> list:
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
