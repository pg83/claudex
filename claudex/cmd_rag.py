import sys
import argparse

import claudex.common as cx
import claudex.rag as rag_mod


RESET = "\033[0m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


SOURCE_COLORS = {"files": CYAN, "conversations": YELLOW}


def cmd_rag(args: argparse.Namespace):
    config = cx.load_config(args.config)
    rag = rag_mod.MultiRAG(config.get("rag", {}))

    print(f"RAG files: {len(rag.files.cache)} / conversations: {len(rag.conversations.cache)}", file=sys.stderr)
    print("> ", end="", file=sys.stderr, flush=True)

    for line in sys.stdin:
        query = line.strip()

        if not query:
            print("> ", end="", file=sys.stderr, flush=True)
            continue

        hits = rag.search(query, 10)

        for h in hits:
            color = SOURCE_COLORS.get(h["source"], "")
            paths = ", ".join(f"{color}{p}{RESET}" for p in h["paths"])
            print(f"{DIM}[{h['rank']:.3f}]{RESET} {paths}")
            print(h["data"])
            print(f"{DIM}---{RESET}")

        print("> ", end="", file=sys.stderr, flush=True)
