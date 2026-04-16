"""models subcommand: list models at an endpoint."""

import argparse
import sys

import httpx

import claudex.common as cx


def cmd_models(args: argparse.Namespace):
    base_url = args.base_url
    api_key = cx._expand_env(args.api_key)

    url = cx._strip_chat_suffix(base_url) + "/models"
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
