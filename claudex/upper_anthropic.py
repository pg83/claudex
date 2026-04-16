import httpx

from typing import AsyncIterator


HOP_BY_HOP = frozenset({"host", "content-length", "connection", "transfer-encoding", "accept-encoding"})


def upstream_headers(client_headers, ep: dict) -> dict:
    out = {k: v for k, v in dict(client_headers).items() if k.lower() not in HOP_BY_HOP}

    if ep["api_key"]:
        out["x-api-key"] = ep["api_key"]
        out.pop("authorization", None)
        out.pop("Authorization", None)

    return out


class AnthropicUpper:
    def __init__(self, server, ep: dict):
        self.server = server
        self.ep = ep

    async def _send(self, body: dict, headers: dict, stream: bool) -> httpx.Response:
        url = self.ep["base_url"].rstrip("/") + "/messages"
        http = self.server.client(self.ep.get("proxy"))
        req = http.build_request("POST", url, json={**body, "stream": stream}, headers=headers)
        resp = await http.send(req, stream=stream)

        if resp.status_code != 200:
            await resp.aread()

            await resp.aclose()

            raise httpx.HTTPStatusError(f"Anthropic {resp.status_code}", request=req, response=resp)

        return resp

    async def call(self, body: dict, headers: dict, req_id: str) -> dict:
        resp = await self._send(body, headers, stream=False)

        return resp.json()

    async def stream(self, body: dict, headers: dict, req_id: str) -> AsyncIterator[str]:
        resp = await self._send(body, headers, stream=True)

        return _iter_chunks(resp)


async def _iter_chunks(resp: httpx.Response) -> AsyncIterator[str]:
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk.decode("utf-8", errors="replace")
    finally:
        await resp.aclose()
