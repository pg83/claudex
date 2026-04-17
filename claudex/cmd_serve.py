import os
import sys
import json
import httpx
import signal
import hashlib
import uvicorn
import argparse


uvicorn.Server.install_signal_handlers = lambda self: None

from contextlib import asynccontextmanager

from typing import AsyncIterator, Optional

from starlette.routing import Route
from starlette.requests import Request
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse

import claudex.log as lg
import claudex.common as cx
import claudex.upper_openai as uo
import claudex.upper_anthropic as ua


# ---------------------------------------------------------------------------
# Proxy-handled tools
# ---------------------------------------------------------------------------

PROXY_HANDLED_TOOLS = {"WebFetch", "WebSearch", "web_search"}

FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def session_id(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user":
            text = cx.extract_text_content(m.get("content", ""))

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


async def debug_stream_wrap(gen: AsyncIterator[str], config: dict, req_id: str) -> AsyncIterator[str]:
    async for event_str in gen:
        lg.debug_sse(config, "out", event_str, req_id=req_id)

        yield event_str


# ---------------------------------------------------------------------------
# ProxyServer: owns http clients, route handlers, lower session
# ---------------------------------------------------------------------------


class ProxyServer:
    def __init__(self, config: dict):
        self.config = config
        self.clients: dict[Optional[str], httpx.AsyncClient] = {}
        self.app = Starlette(
            lifespan=self.lifespan,
            routes=[
                Route("/v1/messages", self.create_message, methods=["POST"]),
                Route("/v1/messages/count_tokens", self.count_tokens, methods=["POST"]),
                Route("/v1/models", self.list_models, methods=["GET"]),
            ],
        )

    def client(self, proxy: Optional[str] = None) -> httpx.AsyncClient:
        if proxy not in self.clients:
            self.clients[proxy] = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=10.0),
                verify=False,
                proxy=proxy,
            )

        return self.clients[proxy]

    @asynccontextmanager
    async def lifespan(self, app: Starlette):
        signal.signal(signal.SIGINT, lambda *_: os._exit(130))
        signal.signal(signal.SIGTERM, lambda *_: os._exit(130))

        yield

        for c in self.clients.values():
            await c.aclose()

    # ----- proxy-handled tools -----

    async def execute_proxy_tool(self, name: str, tool_input: dict, sid: str = "", req_id: str = "") -> str:
        if name == "WebFetch":
            url = tool_input.get("url", "")
            lg.log(f"web_fetch {url}", sid=sid)
            resp = await self.client().get(url, headers=FETCH_HEADERS, timeout=15, follow_redirects=True)
            result = resp.text
            lg.debug_log(self.config, "TOOL EXECUTE", {"name": name, "input": tool_input, "output": result}, req_id=req_id)

            return result

        if name in ("WebSearch", "web_search"):
            query = tool_input.get("query", "")
            lg.log(f"web_search {query}", sid=sid)

            resp = await self.client().get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=FETCH_HEADERS,
                timeout=15,
                follow_redirects=True,
            )

            result = resp.text
            lg.debug_log(self.config, "TOOL EXECUTE", {"name": name, "input": tool_input, "output": result}, req_id=req_id)

            return result

        return f"Unknown tool: {name}"

    # ----- HTTP endpoints -----

    async def create_message(self, request: Request):
        req_id = lg.next_req_id()

        try:
            body = await request.json()
        except Exception:
            return cx.error_response(400, "invalid_request_error", "Invalid JSON body")

        sid = session_id(body.get("messages", []))
        anthropic_model = body.get("model", "")
        is_stream = body.get("stream", False)
        ep = cx.resolve_endpoint(self.config, anthropic_model)
        n_msgs = len(body.get("messages", []))
        n_tools = len(body.get("tools", []))
        body_bytes = len(json.dumps(body))
        stream_tag = "stream" if is_stream else "sync"

        lg.log(f"{anthropic_model} -> {ep['model']} | {n_msgs} msgs, {n_tools} tools, ~{body_bytes//4}tok, {lg.human_bytes(body_bytes)}, {stream_tag}",
            sid=sid)

        lg.debug_log(self.config, "ANTHROPIC REQUEST", body, req_id=req_id,
                  sid=sid, model=anthropic_model, stream=is_stream,
                  endpoint_model=ep["model"],
                  messages=n_msgs, tools=n_tools)

        body["model"] = ep["model"]

        if ep.get("protocol") == "anthropic":
            upper = ua.AnthropicUpper(self, ep)
        else:
            upper = uo.OpenAIUpper(self, ep)

        try:
            return await self._lower_handle(upper, body, anthropic_model, sid, req_id, request.headers, is_stream)
        except httpx.HTTPStatusError as e:
            try:
                error_body = e.response.json()
            except Exception:
                error_body = None

            lg.log(f"ERROR {e.response.status_code}", sid=sid)
            lg.debug_log(self.config, "UPSTREAM ERROR", error_body, req_id=req_id, sid=sid, status=e.response.status_code)

            return upper.translate_error(e.response.status_code, error_body)

        except Exception as e:
            lg.log(f"ERROR {e}", sid=sid)
            lg.debug_log(self.config, "PROXY ERROR", {"error": str(e)}, req_id=req_id, sid=sid)

            return cx.error_response(500, "api_error", str(e))

    async def _lower_handle(self, upper, body: dict, anthropic_model: str, sid: str, req_id: str, client_headers, is_stream: bool):
        proto = type(upper).__name__

        if is_stream:
            lg.debug_log(self.config, "UPSTREAM REQUEST", body, req_id=req_id, sid=sid, stream=True, proto=proto,
                      endpoint_model=body.get("model", ""), anthropic_model=anthropic_model)
            iterator = await upper.stream(body, client_headers, sid)

            return StreamingResponse(
                debug_stream_wrap(iterator, self.config, req_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        resp: dict = {}

        for _ in range(6):
            lg.debug_log(self.config, "UPSTREAM REQUEST", body, req_id=req_id, sid=sid, proto=proto,
                      endpoint_model=body.get("model", ""), anthropic_model=anthropic_model,
                      messages=len(body.get("messages", [])))
            resp = await upper.call(body, client_headers, sid)
            lg.debug_log(self.config, "UPSTREAM RESPONSE", resp, req_id=req_id, sid=sid, proto=proto,
                      stop=resp.get("stop_reason"))

            proxy_uses = extract_proxy_tool_uses(resp)

            if not proxy_uses:
                break

            body["messages"].append({"role": "assistant", "content": resp.get("content", [])})

            tool_results = []

            for tu in proxy_uses:
                try:
                    result = await self.execute_proxy_tool(tu["name"], tu.get("input", {}), sid=sid, req_id=req_id)
                except Exception as e:
                    result = f"Error: {e}"
                    lg.log(f"tool error {tu['name']}: {e}", sid=sid)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result,
                })

            body["messages"].append({"role": "user", "content": tool_results})

        resp["model"] = anthropic_model

        usage = resp.get("usage", {})
        lg.log(f"done | {usage.get('input_tokens',0)}in/{usage.get('output_tokens',0)}out | {resp.get('stop_reason','')}",
            sid=sid)
        lg.debug_log(self.config, "ANTHROPIC RESPONSE", resp, req_id=req_id, sid=sid,
                  stop=resp.get("stop_reason"))

        return JSONResponse(content=resp)

    async def count_tokens(self, request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"input_tokens": 0})

        total_chars = len(cx.extract_system_text(body.get("system")))

        for msg in body.get("messages", []):
            total_chars += count_content_chars(msg.get("content", ""))

        if "tools" in body:
            total_chars += len(json.dumps(body["tools"]))

        return JSONResponse(content={"input_tokens": max(1, total_chars // 4)})

    async def list_models(self, request: Request):
        endpoints = self.config.get("endpoints", {})
        models = [f"claude-{role}" for role in endpoints] or [
            "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5",
        ]

        return JSONResponse(content={
            "data": [
                {"id": m, "type": "model", "display_name": m, "created_at": "2025-01-01T00:00:00Z"}
                for m in models
            ],
            "has_more": False,
        })

    def run(self):
        host = self.config.get("host", "127.0.0.1")
        port = self.config.get("port", 8082)
        uvicorn.run(self.app, host=host, port=port, log_level="info", access_log=False)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace):
    config = cx.load_config(args.config)

    if args.debug:
        config["debug"] = True

    if args.port:
        config["port"] = args.port

    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8082)
    endpoints = config.get("endpoints", {})

    def info(msg):
        print(msg, file=sys.stderr, flush=True)

    info(f"Proxy starting on {host}:{port}")
    info("Endpoints:")

    for role, ep in endpoints.items():
        info(f"  {role} [{ep['protocol']}]: {ep['base_url']} -> {ep['model']}")

    if config["debug"]:
        info("Debug: ENABLED (JSONL to stdout, redirect with > debug.jsonl)")

    info("")
    info("Usage:")
    info(f"  ANTHROPIC_BASE_URL=http://{host}:{port} ANTHROPIC_API_KEY=dummy claude")

    server = ProxyServer(config)
    server.run()
