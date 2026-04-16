import httpx

from typing import AsyncIterator, Optional

import claudex.log as lg
import claudex.proto_common as pc


def messages_url(base_url: str) -> str:
    base = base_url.rstrip("/")

    if base.endswith("/messages"):
        return base

    return base + "/messages"


def anthropic_headers(api_key: str, client_headers) -> dict:
    headers = {"content-type": "application/json"}

    if client_headers:
        for h in pc.FORWARD_HEADERS:
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

    return headers


class AnthropicUpper:
    def __init__(self, server, ep: dict):
        self.server = server
        self.ep = ep

    def _upstream_body(self, body: dict, stream: bool) -> dict:
        target = dict(body)
        target["messages"] = list(body.get("messages", []))
        target["model"] = self.ep["model"]
        target["stream"] = stream

        return target

    async def call(self, body: dict, client_headers, req_id: str) -> dict:
        target = self._upstream_body(body, stream=False)

        lg.debug_log(self.server.config, "ANTHROPIC UPSTREAM REQUEST", target, req_id=req_id,
                  model=target.get("model", ""),
                  base_url=self.ep["base_url"],
                  messages=len(target.get("messages", [])))

        headers = anthropic_headers(self.ep["api_key"], client_headers)
        url = messages_url(self.ep["base_url"])

        resp = await self.server.client(self.ep.get("proxy")).post(url, json=target, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        lg.debug_log(self.server.config, "ANTHROPIC UPSTREAM RESPONSE", data, req_id=req_id,
                  stop=data.get("stop_reason"))

        return data

    async def stream(self, body: dict, client_headers, req_id: str) -> AsyncIterator[str]:
        target = self._upstream_body(body, stream=True)

        lg.debug_log(self.server.config, "ANTHROPIC UPSTREAM REQUEST", target, req_id=req_id,
                  model=target.get("model", ""),
                  base_url=self.ep["base_url"],
                  messages=len(target.get("messages", [])),
                  stream=True)

        headers = anthropic_headers(self.ep["api_key"], client_headers)
        url = messages_url(self.ep["base_url"])
        http = self.server.client(self.ep.get("proxy"))

        req = http.build_request("POST", url, json=target, headers=headers)
        resp = await http.send(req, stream=True)

        if resp.status_code != 200:
            await resp.aread()

            await resp.aclose()

            raise httpx.HTTPStatusError(
                f"Anthropic returned {resp.status_code}",
                request=req,
                response=resp,
            )

        return _iter_passthrough(resp)


async def _iter_passthrough(resp: httpx.Response) -> AsyncIterator[str]:
    buffer = b""

    try:
        async for chunk in resp.aiter_bytes():
            buffer += chunk

            while b"\n\n" in buffer:
                event_bytes, buffer = buffer.split(b"\n\n", 1)

                yield event_bytes.decode("utf-8", errors="replace") + "\n\n"

        if buffer:
            tail = buffer.decode("utf-8", errors="replace")

            if tail.strip():
                yield tail
    finally:
        await resp.aclose()
