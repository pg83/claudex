import os
import sys
import json
import httpx
import uvicorn
import argparse

from contextlib import asynccontextmanager

from typing import AsyncIterator, Optional

from starlette.routing import Route
from starlette.requests import Request
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse

import claudex.log as lg
import claudex.common as cx
import claudex.rag as rag_mod
import claudex.proto_common as pc
import claudex.upper_openai as uo
import claudex.upper_anthropic as ua


# ---------------------------------------------------------------------------
# ProxyServer: owns http clients, rag, compress, route handlers, lower session
# ---------------------------------------------------------------------------


class ProxyServer:
    def __init__(self, config: dict, rag: Optional[rag_mod.RAG] = None):
        self.config = config
        self.rag = rag
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
        yield

        for c in self.clients.values():
            await c.aclose()

    # ----- proxy-handled tools -----

    async def execute_proxy_tool(self, name: str, tool_input: dict, req_id: str = "") -> str:
        if name == "WebFetch":
            url = tool_input.get("url", "")
            lg.log(f"web_fetch {url}", req_id=req_id)
            resp = await self.client().get(url, headers=pc.FETCH_HEADERS, timeout=15, follow_redirects=True)
            result = resp.text
            lg.debug_log(self.config, "TOOL EXECUTE", {"name": name, "input": tool_input, "output": result}, req_id=req_id)

            return result

        if name in ("WebSearch", "web_search"):
            query = tool_input.get("query", "")
            lg.log(f"web_search {query}", req_id=req_id)

            resp = await self.client().get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=pc.FETCH_HEADERS,
                timeout=15,
                follow_redirects=True,
            )

            result = resp.text
            lg.debug_log(self.config, "TOOL EXECUTE", {"name": name, "input": tool_input, "output": result}, req_id=req_id)

            return result

        return f"Unknown tool: {name}"

    # ----- compression -----

    async def call_compress_llm(self, messages: list, req_id: str = "") -> str:
        ep = cx.resolve_endpoint(self.config, "compress")
        compress_body = pc.build_compress_body(messages, ep["model"])
        lg.debug_log(self.config, "COMPRESS_REQ", compress_body, req_id=req_id)

        headers = {
            "Authorization": f"Bearer {ep['api_key']}",
            "Content-Type": "application/json",
        }
        url = uo.chat_url(ep["base_url"])

        resp = await self.client(ep.get("proxy")).post(url, json=compress_body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        if "response" in data and "choices" not in data:
            data = data["response"]

        lg.debug_log(self.config, "COMPRESS_RESP", data, req_id=req_id)

        return data["choices"][0]["message"]["content"] or ""

    async def compress_context(self, messages: list, req_id: str = "") -> list:
        keep = self.config["compress_keep"]
        min_msgs = self.config["compress_min"]

        if len(messages) < min_msgs:
            return messages

        split = len(messages) - keep

        if split < 2:
            return messages

        to_collapse = messages[:split]
        tail = messages[split:]

        collapsed = pc.collapse_messages(to_collapse)
        compressed = [*collapsed, *tail]

        original_total = len(json.dumps(messages, ensure_ascii=False))
        compressed_bytes = len(json.dumps(compressed, ensure_ascii=False))
        ratio = original_total / compressed_bytes if compressed_bytes else 0

        lg.log(f"compress {len(messages)}→{len(compressed)} msgs, {lg.human_bytes(original_total)}→{lg.human_bytes(compressed_bytes)}, {ratio:.1f}x",
            req_id=req_id)

        lg.debug_log(self.config, "CONTEXT COMPRESSION", compressed, req_id=req_id,
                  original_msgs=len(messages),
                  compressed_to=len(compressed),
                  kept_verbatim=len(tail),
                  original_total=original_total,
                  compressed_bytes=compressed_bytes)

        return compressed

    # ----- RAG enrichment -----

    def enrich_with_rag(self, body: dict, req_id: str):
        rag_cfg = self.config.get("rag", {})
        chunk_size = rag_cfg.get("chunk_size", 2000)

        for msg in body.get("messages", []):
            text = pc.extract_msg_text(msg)

            if text:
                self.rag.add(f"conversation/{req_id}/{msg['role']}", text, chunk_size)

        last_text = pc.extract_last_user_text(body.get("messages", []))

        if not last_text:
            return

        rag_results = self.rag.search(last_text, rag_cfg.get("max_results", 3))

        if not rag_results:
            return

        rag_block = "\n".join(
            f"Files: {', '.join(r['paths'])}\n---\n{r['data']}\n---"
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

        q_preview = last_text[:200].replace("\n", " ")
        lg.log(f"rag query: {q_preview!r}", req_id=req_id)

        for r in rag_results:
            data_preview = r["data"][:200].replace("\n", " ")
            paths = ", ".join(r["paths"])
            lg.log(f"  rag hit: {paths}({r['rank']:.2f}) {data_preview!r}", req_id=req_id)

        lg.debug_log(self.config, "RAG", {"query": last_text, "results": rag_results}, req_id=req_id)

    # ----- HTTP endpoints -----

    async def create_message(self, request: Request):
        req_id = lg.next_req_id()

        try:
            body = await request.json()
        except Exception:
            return pc.error_response(400, "invalid_request_error", "Invalid JSON body")

        anthropic_model = body.get("model", "")
        is_stream = body.get("stream", False)
        ep = cx.resolve_endpoint(self.config, anthropic_model)
        n_msgs = len(body.get("messages", []))
        n_tools = len(body.get("tools", []))
        body_bytes = len(json.dumps(body))
        stream_tag = "stream" if is_stream else "sync"

        lg.log(f"{anthropic_model} -> {ep['model']} | {n_msgs} msgs, {n_tools} tools, ~{body_bytes//4}tok, {lg.human_bytes(body_bytes)}, {stream_tag}",
            req_id=req_id)

        lg.debug_log(self.config, "ANTHROPIC REQUEST", body, req_id=req_id,
                  model=anthropic_model, stream=is_stream,
                  endpoint_model=ep["model"],
                  messages=n_msgs, tools=n_tools)

        if "compress" in self.config["endpoints"] and "messages" in body:
            body["messages"] = await self.compress_context(body["messages"], req_id=req_id)

        if self.rag is not None:
            self.enrich_with_rag(body, req_id)

        if ep.get("protocol") == "anthropic":
            upper = ua.AnthropicUpper(self, ep)
        else:
            upper = uo.OpenAIUpper(self, ep)

        try:
            return await self._lower_handle(upper, body, anthropic_model, req_id, request.headers, is_stream)
        except httpx.HTTPStatusError as e:
            try:
                error_body = e.response.json()
            except Exception:
                error_body = None

            lg.log(f"ERROR {e.response.status_code}", req_id=req_id)
            lg.debug_log(self.config, "UPSTREAM ERROR", error_body, req_id=req_id, status=e.response.status_code)

            return pc.translate_openai_error(e.response.status_code, error_body)

        except Exception as e:
            lg.log(f"ERROR {e}", req_id=req_id)
            lg.debug_log(self.config, "PROXY ERROR", {"error": str(e)}, req_id=req_id)

            return pc.error_response(500, "api_error", str(e))

    async def _lower_handle(self, upper, body: dict, anthropic_model: str, req_id: str, client_headers, is_stream: bool):
        if is_stream:
            iterator = await upper.stream(body, client_headers, req_id)

            return StreamingResponse(
                pc.debug_stream_wrap(iterator, self.config, req_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        resp: dict = {}

        for _ in range(6):
            resp = await upper.call(body, client_headers, req_id)

            proxy_uses = pc.extract_proxy_tool_uses(resp)

            if not proxy_uses:
                break

            body["messages"].append({"role": "assistant", "content": resp.get("content", [])})

            tool_results = []

            for tu in proxy_uses:
                try:
                    result = await self.execute_proxy_tool(tu["name"], tu.get("input", {}), req_id=req_id)
                except Exception as e:
                    result = f"Error: {e}"
                    lg.log(f"tool error {tu['name']}: {e}", req_id=req_id)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result,
                })

            body["messages"].append({"role": "user", "content": tool_results})

        resp["model"] = anthropic_model

        usage = resp.get("usage", {})
        lg.log(f"done | {usage.get('input_tokens',0)}in/{usage.get('output_tokens',0)}out | {resp.get('stop_reason','')}",
            req_id=req_id)
        lg.debug_log(self.config, "ANTHROPIC RESPONSE", resp, req_id=req_id,
                  stop=resp.get("stop_reason"))

        return JSONResponse(content=resp)

    async def count_tokens(self, request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"input_tokens": 0})

        total_chars = len(pc.extract_system_text(body.get("system")))

        for msg in body.get("messages", []):
            total_chars += pc.count_content_chars(msg.get("content", ""))

        if "tools" in body:
            total_chars += len(json.dumps(body["tools"]))

        return JSONResponse(content={"input_tokens": max(1, total_chars // 4)})

    async def list_models(self, request: Request):
        endpoints = self.config.get("endpoints", {})
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

    def run(self):
        host = self.config.get("host", "127.0.0.1")
        port = self.config.get("port", 8082)
        uvicorn.run(self.app, host=host, port=port, log_level="info")


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

    if "compress" in endpoints:
        info(f"Compression: keep={config['compress_keep']}, min={config['compress_min']}")

    rag = None
    rag_cfg = config.get("rag", {})
    rag_dirs = rag_cfg.get("dirs", [])

    if rag_dirs:
        dirs = [os.path.expanduser(d) for d in rag_dirs]
        exts = rag_cfg.get("extensions")
        chunk_size = rag_cfg.get("chunk_size", 2000)
        db_path = rag_cfg.get("db", "~/.cache/claudex/rag.db")

        embed_url = rag_cfg.get("embed_url") or rag_mod.OLLAMA_URL
        embed_model = rag_cfg.get("embed_model") or rag_mod.OLLAMA_MODEL
        embedder = rag_mod.make_ollama_embedder(embed_url, embed_model)

        rag = rag_mod.RAG(db_path, dirs, set(exts) if exts else None, chunk_size, embedder)
        info(f"RAG: {len(rag.cache)} chunks from {', '.join(dirs)} (db: {db_path}, embed: {embed_model} @ {embed_url})")

    if config["debug"]:
        info("Debug: ENABLED (JSONL to stdout, redirect with > debug.jsonl)")

    info("")
    info("Usage:")
    info(f"  ANTHROPIC_BASE_URL=http://{host}:{port} ANTHROPIC_API_KEY=dummy claude")

    server = ProxyServer(config, rag)
    server.run()
