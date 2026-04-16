#!/usr/bin/env python3
"""Claudex proxy CLI entry point.

Shared config lives in common.py; each subcommand owns its own
cmd_*.py module.
"""

import argparse


def main():
    from cmd_anal import cmd_anal
    from cmd_models import cmd_models
    from cmd_serve import cmd_serve

    parser = argparse.ArgumentParser(description="Claudex proxy")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start the proxy server")
    p_serve.add_argument("config", help="Path to config.json")
    p_serve.add_argument("--debug", action="store_true", help="Enable debug logging")
    p_serve.add_argument("--port", type=int, default=0, help="Override listen port")

    p_models = sub.add_parser("models", help="List models at an endpoint")
    p_models.add_argument("config", nargs="?", help="Path to config.json (uses first endpoint)")
    p_models.add_argument("--base-url", help="Base URL (e.g. https://api.openai.com/v1)")
    p_models.add_argument("--api-key", help="API key (or $ENV_VAR)")

    p_anal = sub.add_parser("anal", help="Analyze debug log")
    p_anal.add_argument("log", help="Path to debug log file")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "anal":
        cmd_anal(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
