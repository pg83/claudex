"""Shared config loading and endpoint resolution used by every subcommand."""

import os
import re
import json
import random

from urllib.parse import urlparse

from starlette.responses import JSONResponse


FALLBACK_CHAIN = {
    "haiku": "sonnet",
    "sonnet": "opus",
}


def _expand_env(s: str) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', lambda m: os.environ.get(m.group(1), m.group(0)), s)


_ANTHROPIC_HOSTS = ("anthropic.com",)


def _infer_protocol(base_url: str) -> str:
    host = (urlparse(base_url).hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in _ANTHROPIC_HOSTS):
        return "anthropic"
    return "openai"


def _strip_chat_suffix(url: str) -> str:
    base = url.rstrip("/")
    for suffix in ("/chat/completions", "/chat/completion"):
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip("/")
    return base


def load_config(path: str) -> dict:
    """Load JSON config file and return a config dict with nested endpoints."""
    with open(path) as f:
        raw = f.read()

    raw = re.sub(r'\$RANDOM', lambda _: str(random.randint(0, 32767)), raw)
    cfg = json.loads(raw)

    shared_api_key = _expand_env(cfg.get("api_key", ""))

    config: dict = {
        "debug": cfg.get("debug", False),
    }

    listen = cfg.get("listen", "127.0.0.1:8082")

    if ":" in listen:
        host, port = listen.rsplit(":", 1)
        config["host"] = host
        config["port"] = int(port)
    else:
        config["host"] = "127.0.0.1"
        config["port"] = int(listen)

    config["search"] = cfg.get("search", {})

    endpoints: dict = {}

    for role, ep in cfg.get("endpoints", {}).items():
        base_url = _expand_env(ep.get("base_url", ""))
        endpoints[role] = {
            "base_url": base_url,
            "model": _expand_env(ep.get("model", "")),
            "api_key": _expand_env(ep.get("api_key", shared_api_key)),
            "ssl_verify": ep.get("ssl_verify", False),
            "protocol": ep.get("protocol") or _infer_protocol(base_url),
            "proxy": _expand_env(ep.get("proxy")) if ep.get("proxy") else None,
        }

    config["endpoints"] = endpoints

    return config


def error_response(status_code: int, etype: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={
        "type": "error",
        "error": {"type": etype, "message": message},
    })


def extract_text_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]

        return " ".join(texts)

    return ""


def extract_system_text(system) -> str:
    if isinstance(system, str):
        return system

    if isinstance(system, list):
        return "\n\n".join(
            block["text"] for block in system if block.get("type") == "text"
        )

    return str(system) if system else ""


def resolve_endpoint(config: dict, name: str) -> dict:
    """Resolve a role name or Anthropic model name to an endpoint dict.

    Fallback chain (hardcoded): haiku -> sonnet -> opus.
    """
    endpoints = config.get("endpoints", {})

    if name in endpoints:
        return endpoints[name]

    role = None
    name_lower = name.lower()

    if "opus" in name_lower:
        role = "opus"
    elif "haiku" in name_lower:
        role = "haiku"
    elif "sonnet" in name_lower:
        role = "sonnet"

    current = role
    visited = set()

    while current:
        if current in endpoints:
            return endpoints[current]

        if current in visited:
            break

        visited.add(current)
        current = FALLBACK_CHAIN.get(current)

    if endpoints:
        return next(iter(endpoints.values()))

    return {"base_url": "", "model": name, "api_key": "", "ssl_verify": False, "protocol": "openai"}
