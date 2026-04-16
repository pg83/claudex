"""models subcommand: list models at an endpoint."""

import argparse
import sys

import httpx

from proxy import ENDPOINTS, _expand_env, _strip_chat_suffix, load_config


def cmd_models(args: argparse.Namespace):
    if args.base_url:
        base_url = args.base_url
        api_key = _expand_env(args.api_key or "")
    elif args.config:
        load_config(args.config)
        if not ENDPOINTS:
            sys.exit("No endpoints configured")
        ep = next(iter(ENDPOINTS.values()))
        base_url = ep["base_url"]
        api_key = _expand_env(args.api_key) if args.api_key else ep["api_key"]
    else:
        sys.exit("Provide either config or --base-url")

    url = _strip_chat_suffix(base_url) + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Fetching models from {url} ...\n")
    try:
        with httpx.Client(verify=False, timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        sys.exit(f"HTTP {e.response.status_code}: {e.response.text}")
    except Exception as e:
        sys.exit(f"Error: {e}")

    if "response" in data and "data" not in data:
        data = data["response"]

    models = data.get("data", [])
    if not models:
        print("No models returned.")
        return

    models.sort(key=lambda m: m.get("id", ""))
    print(f"Found {len(models)} models:\n")
    for m in models:
        mid = m.get("id", "?")
        owned_by = m.get("owned_by", "")
        extra = f"  (by {owned_by})" if owned_by else ""
        print(f"  {mid}{extra}")
