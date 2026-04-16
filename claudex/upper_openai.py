import json
import uuid
import httpx

from typing import AsyncIterator, Optional, Union

import claudex.log as lg
import claudex.common as cx
import claudex.proto_common as pc


def chat_url(base_url: str) -> str:
    return cx._strip_chat_suffix(base_url) + "/chat/completions"


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


TOOL_CHOICE_MAP = {"auto": "auto", "any": "required", "none": "none"}


def convert_tool_choice(anthropic_tc) -> Union[str, dict]:
    if not isinstance(anthropic_tc, dict):
        return "auto"

    tc_type = anthropic_tc.get("type", "auto")

    if tc_type == "tool":
        return {"type": "function", "function": {"name": anthropic_tc["name"]}}

    return TOOL_CHOICE_MAP.get(tc_type, "auto")


def convert_user_msg(content) -> list[dict]:
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
            "tool_call_id": pc.to_openai_tool_id(tr["tool_use_id"]),
            "content": extract_tool_result_content(tr),
        })

    if other_blocks:
        converted = convert_content_to_openai(other_blocks)

        if converted:
            result.append({"role": "user", "content": converted})

    return result


def convert_assistant_msg(content) -> dict:
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
                    "id": pc.to_openai_tool_id(block["id"]),
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

    system_text = pc.extract_system_text(body.get("system"))

    if system_text:
        openai_messages.append({"role": "developer", "content": system_text})

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            openai_messages.extend(convert_user_msg(content))
        elif role == "assistant":
            openai_messages.append(convert_assistant_msg(content))

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
                "id": pc.to_anthropic_tool_id(tc["id"]),
                "name": tc["function"]["name"],
                "input": tool_input,
            })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop_reason = FINISH_REASON_MAP.get(choice.get("finish_reason"), "end_turn")
    usage = openai_resp.get("usage", {})

    return {
        "id": pc.gen_message_id(),
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
# Streaming translation: OpenAI SSE -> Anthropic SSE
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
        self.message_id = pc.gen_message_id()
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
        events.append(pc.sse_event("content_block_delta", {
            "index": state.block_index,
            "delta": {"type": "signature_delta", "signature": ""},
        }))

    events.append(pc.sse_event("content_block_stop", {
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

    events.append(pc.sse_event("content_block_start", {
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

    yield pc.sse_event("message_start", {
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

    yield pc.sse_event("ping", {})

    finish_reason = None
    chunk_count = 0
    raw_chunks = []

    try:
        async for chunk in iter_openai_sse(openai_response):
            if chunk == "[DONE]":
                lg.debug_sse(config, "in", "event: done\ndata: [DONE]\n\n", req_id=req_id)

                break

            chunk_count += 1
            raw_chunks.append(chunk)
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

                    yield pc.sse_event("content_block_delta", {
                        "index": state.block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    })

                text = delta.get("content")

                if text:
                    for ev in open_block_events(state, "text", {"type": "text", "text": ""}):
                        yield ev

                    yield pc.sse_event("content_block_delta", {
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

                            tool_id = pc.to_anthropic_tool_id(tc_delta.get("id", f"call_{uuid.uuid4().hex[:8]}"))
                            tool_name = tc_delta.get("function", {}).get("name", "")
                            state.current_block_type = "tool_use"
                            state.current_tool_index = tc_idx

                            yield pc.sse_event("content_block_start", {
                                "index": state.block_index,
                                "content_block": {
                                    "type": "tool_use", "id": tool_id, "name": tool_name, "input": {},
                                },
                            })

                        args_chunk = tc_delta.get("function", {}).get("arguments", "")

                        if args_chunk:
                            yield pc.sse_event("content_block_delta", {
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

    yield pc.sse_event("message_delta", {
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": state.output_tokens},
    })

    yield pc.sse_event("message_stop", {})

    lg.log(f"done | {state.input_tokens}in/{state.output_tokens}out | {stop_reason}",
        req_id=req_id)
    lg.debug_log(config, "OPENAI RESPONSE (stream)", raw_chunks, req_id=req_id,
              chunks=chunk_count,
              stop=stop_reason,
              input_tokens=state.input_tokens,
              output_tokens=state.output_tokens)


# ---------------------------------------------------------------------------
# Upper session
# ---------------------------------------------------------------------------


class OpenAIUpper:
    def __init__(self, server, ep: dict):
        self.server = server
        self.ep = ep

    async def _post(self, openai_body: dict, stream: bool, client_headers) -> Union[httpx.Response, dict]:
        headers = {"Content-Type": "application/json"}

        if self.ep["api_key"]:
            headers["Authorization"] = f"Bearer {self.ep['api_key']}"
        elif client_headers:
            auth = client_headers.get("authorization")

            if auth:
                headers["Authorization"] = auth

        url = chat_url(self.ep["base_url"])
        http = self.server.client(self.ep.get("proxy"))

        if stream:
            req = http.build_request("POST", url, json=openai_body, headers=headers)
            resp = await http.send(req, stream=True)

            if resp.status_code != 200:
                await resp.aread()

                await resp.aclose()

                raise httpx.HTTPStatusError(
                    f"OpenAI returned {resp.status_code}",
                    request=req,
                    response=resp,
                )

            return resp

        resp = await http.post(url, json=openai_body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        if "response" in data and "choices" not in data:
            data = data["response"]

        return data

    async def call(self, body: dict, client_headers, req_id: str) -> dict:
        anthropic_model = body.get("model", "")
        openai_body = convert_request(body, self.ep["model"])
        openai_body["stream"] = False
        openai_body.pop("stream_options", None)

        lg.debug_log(self.server.config, "OPENAI REQUEST", openai_body, req_id=req_id,
                  model=openai_body.get("model", ""),
                  base_url=self.ep["base_url"],
                  messages=len(openai_body.get("messages", [])))

        openai_resp = await self._post(openai_body, stream=False, client_headers=client_headers)

        lg.debug_log(self.server.config, "OPENAI RESPONSE", openai_resp, req_id=req_id,
                  finish=openai_resp.get("choices", [{}])[0].get("finish_reason"))

        return convert_response(openai_resp, anthropic_model)

    async def stream(self, body: dict, client_headers, req_id: str) -> AsyncIterator[str]:
        anthropic_model = body.get("model", "")
        openai_body = convert_request(body, self.ep["model"])
        openai_body["stream"] = True
        openai_body["stream_options"] = {"include_usage": True}

        lg.debug_log(self.server.config, "OPENAI REQUEST", openai_body, req_id=req_id,
                  model=openai_body.get("model", ""),
                  base_url=self.ep["base_url"],
                  messages=len(openai_body.get("messages", [])),
                  stream=True)

        resp = await self._post(openai_body, stream=True, client_headers=client_headers)

        return stream_translate(resp, anthropic_model, self.server.config, req_id=req_id)
