"""models subcommand: list models at an endpoint."""

import httpx
import argparse

import claudex.common as cx


def cmd_models(args: argparse.Namespace):
    base_url = args.base_url
    api_key = cx._expand_env(args.api_key)

    url = cx._strip_chat_suffix(base_url) + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Fetching models from {url} ...\n")

    with httpx.Client(verify=False, timeout=30.0) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

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
