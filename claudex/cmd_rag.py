import os
import sys
import argparse

import claudex.common as cx
import claudex.rag as rag_mod


RESET = "\033[0m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


def color_path(path: str) -> str:
    if path.startswith("conversation/"):
        return f"{YELLOW}{path}{RESET}"

    return f"{CYAN}{path}{RESET}"


def cmd_rag(args: argparse.Namespace):
    config = cx.load_config(args.config)
    rag_cfg = config.get("rag", {})

    dirs = [os.path.expanduser(d) for d in rag_cfg.get("dirs", [])]
    exts = rag_cfg.get("extensions")
    chunk_size = rag_cfg.get("chunk_size", 2000)
    db_path = rag_cfg.get("db", "~/.cache/claudex/rag.db")

    embed_url = rag_cfg.get("embed_url") or rag_mod.OLLAMA_URL
    embed_model = rag_cfg.get("embed_model") or rag_mod.OLLAMA_MODEL
    embedder = rag_mod.make_ollama_embedder(embed_url, embed_model)

    rag = rag_mod.RAG(db_path, dirs, set(exts) if exts else None, chunk_size, embedder)

    print(f"RAG: {len(rag.cache)} chunks (db: {db_path}, embed: {embed_model} @ {embed_url})", file=sys.stderr)
    print("> ", end="", file=sys.stderr, flush=True)

    for line in sys.stdin:
        query = line.strip()

        if not query:
            print("> ", end="", file=sys.stderr, flush=True)
            continue

        hits = rag.search(query, 10)

        for h in hits:
            paths = ", ".join(color_path(p) for p in h["paths"])
            print(f"{DIM}[{h['rank']:.3f}]{RESET} {paths}")
            print(h["data"])
            print(f"{DIM}---{RESET}")

        print("> ", end="", file=sys.stderr, flush=True)
