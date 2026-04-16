"""Shared config loading and endpoint resolution used by every subcommand."""

import json
import os
import re
from urllib.parse import urlparse


CONFIG = {
    "debug": False,
    "compress_keep": 4,
    "compress_min": 8,
}

# role -> {base_url, model, api_key, ssl_verify, protocol}
ENDPOINTS: dict = {}

FALLBACK_CHAIN = {
    "compress": "haiku",
    "haiku": "sonnet",
    "sonnet": "opus",
}


def _expand_env(s: str) -> str:
    """Expand $ENV_VAR references in a string."""
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
    """Strip /chat/completions or /chat/completion to get the API root."""
    base = url.rstrip("/")
    for suffix in ("/chat/completions", "/chat/completion"):
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip("/")
    return base


def load_config(path: str):
    """Load JSON config file, populate CONFIG and ENDPOINTS."""
    with open(path) as f:
        cfg = json.load(f)

    shared_api_key = _expand_env(cfg.get("api_key", ""))

    listen = cfg.get("listen", "127.0.0.1:8082")
    if ":" in listen:
        host, port = listen.rsplit(":", 1)
        CONFIG["host"] = host
        CONFIG["port"] = int(port)
    else:
        CONFIG["host"] = "127.0.0.1"
        CONFIG["port"] = int(listen)

    CONFIG["compress_keep"] = cfg.get("compress_keep", 4)
    CONFIG["compress_min"] = cfg.get("compress_min", 8)
    CONFIG["debug"] = cfg.get("debug", False)

    raw = cfg.get("rag_dir")
    if isinstance(raw, str):
        CONFIG["rag_dirs"] = [raw]
    elif isinstance(raw, list):
        CONFIG["rag_dirs"] = raw
    else:
        CONFIG["rag_dirs"] = []
    CONFIG["rag_extensions"] = cfg.get("rag_extensions")
    CONFIG["rag_max_results"] = cfg.get("rag_max_results", 3)
    CONFIG["rag_chunk_size"] = cfg.get("rag_chunk_size", 2000)

    for role, ep in cfg.get("endpoints", {}).items():
        base_url = _expand_env(ep.get("base_url", ""))
        ENDPOINTS[role] = {
            "base_url": base_url,
            "model": _expand_env(ep.get("model", "")),
            "api_key": _expand_env(ep.get("api_key", shared_api_key)),
            "ssl_verify": ep.get("ssl_verify", False),
            "protocol": ep.get("protocol") or _infer_protocol(base_url),
        }


def resolve_endpoint(name: str) -> dict:
    """Resolve a role name or Anthropic model name to an endpoint dict.

    Fallback chain (hardcoded): compress -> haiku -> sonnet -> opus.
    """
    if name in ENDPOINTS:
        return ENDPOINTS[name]

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
        if current in ENDPOINTS:
            return ENDPOINTS[current]
        if current in visited:
            break
        visited.add(current)
        current = FALLBACK_CHAIN.get(current)

    if ENDPOINTS:
        return next(iter(ENDPOINTS.values()))
    return {"base_url": "", "model": name, "api_key": "", "ssl_verify": False, "protocol": "openai"}
