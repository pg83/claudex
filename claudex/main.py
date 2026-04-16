"""Claudex proxy CLI dispatcher."""

import argparse

import claudex.cmd_anal as ca
import claudex.cmd_models as cm
import claudex.cmd_serve as cs


def main():
    parser = argparse.ArgumentParser(description="Claudex proxy")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start the proxy server")
    p_serve.add_argument("config", help="Path to config.json")
    p_serve.add_argument("--debug", action="store_true", help="Enable debug logging")
    p_serve.add_argument("--port", type=int, default=0, help="Override listen port")

    p_models = sub.add_parser("models", help="List models at an endpoint")
    p_models.add_argument("--base-url", required=True, help="Base URL (e.g. https://api.openai.com/v1)")
    p_models.add_argument("--api-key", required=True, help="API key (or $ENV_VAR)")

    p_anal = sub.add_parser("anal", help="Analyze debug log")
    p_anal.add_argument("log", help="Path to debug log file")

    args = parser.parse_args()

    if args.command == "serve":
        cs.cmd_serve(args)
    elif args.command == "models":
        cm.cmd_models(args)
    elif args.command == "anal":
        ca.cmd_anal(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
