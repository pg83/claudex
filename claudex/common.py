"""Shared config loading and endpoint resolution used by every subcommand."""

import os
import re
import json

from urllib.parse import urlparse


FALLBACK_CHAIN = {
    "compress": "haiku",
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
        cfg = json.load(f)

    shared_api_key = _expand_env(cfg.get("api_key", ""))

    config: dict = {
        "debug": cfg.get("debug", False),
        "compress_keep": cfg.get("compress_keep", 4),
        "compress_min": cfg.get("compress_min", 8),
    }

    listen = cfg.get("listen", "127.0.0.1:8082")

    if ":" in listen:
        host, port = listen.rsplit(":", 1)
        config["host"] = host
        config["port"] = int(port)
    else:
        config["host"] = "127.0.0.1"
        config["port"] = int(listen)

    raw = cfg.get("rag_dir")

    if isinstance(raw, str):
        config["rag_dirs"] = [raw]
    elif isinstance(raw, list):
        config["rag_dirs"] = raw
    else:
        config["rag_dirs"] = []

    config["rag_extensions"] = cfg.get("rag_extensions")
    config["rag_max_results"] = cfg.get("rag_max_results", 3)
    config["rag_chunk_size"] = cfg.get("rag_chunk_size", 2000)

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


def resolve_endpoint(config: dict, name: str) -> dict:
    """Resolve a role name or Anthropic model name to an endpoint dict.

    Fallback chain (hardcoded): compress -> haiku -> sonnet -> opus.
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
