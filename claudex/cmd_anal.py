"""anal subcommand: pretty-print a debug JSONL log."""

import argparse
import json


def cmd_anal(args: argparse.Namespace):
    for line in open(args.log):
        line = line.strip()

        if not line.startswith("{"):
            continue

        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue

        print(json.dumps(ev, indent=2, ensure_ascii=False))
        print()
