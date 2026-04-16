import httpx

from typing import AsyncIterator

import claudex.proto_common as pc


class AnthropicUpper:
    def __init__(self, server, ep: dict):
        self.server = server
        self.ep = ep

    def _headers(self, client_headers) -> dict:
        out = {"content-type": "application/json", "anthropic-version": "2023-06-01"}

        if client_headers:
            for h in pc.FORWARD_HEADERS:
                v = client_headers.get(h)

                if v:
                    out[h] = v

        if self.ep["api_key"]:
            out["x-api-key"] = self.ep["api_key"]
        elif client_headers:
            for h in ("authorization", "x-api-key"):
                v = client_headers.get(h)

                if v:
                    out[h] = v

        return out

    async def _send(self, body: dict, client_headers, stream: bool) -> httpx.Response:
        url = self.ep["base_url"].rstrip("/") + "/messages"
        http = self.server.client(self.ep.get("proxy"))
        req = http.build_request("POST", url, json={**body, "stream": stream}, headers=self._headers(client_headers))
        resp = await http.send(req, stream=stream)

        if resp.status_code != 200:
            await resp.aread()

            await resp.aclose()

            raise httpx.HTTPStatusError(f"Anthropic {resp.status_code}", request=req, response=resp)

        return resp

    async def call(self, body: dict, client_headers, req_id: str) -> dict:
        resp = await self._send(body, client_headers, stream=False)

        return resp.json()

    async def stream(self, body: dict, client_headers, req_id: str) -> AsyncIterator[str]:
        resp = await self._send(body, client_headers, stream=True)

        return _iter_chunks(resp)


async def _iter_chunks(resp: httpx.Response) -> AsyncIterator[str]:
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk.decode("utf-8", errors="replace")
    finally:
        await resp.aclose()
