import sys
import argparse

import claudex.common as cx
import claudex.search as search_mod


RESET = "\033[0m"
DIM = "\033[2m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
GREEN = "\033[32m"


SOURCE_COLORS = {
    "fs": CYAN,
    "conversation": YELLOW,
}


ENGINE_COLORS = {
    "rag": GREEN,
    "whoosh": MAGENTA,
}


def cmd_search(args: argparse.Namespace):
    config = cx.load_config(args.config)
    s = search_mod.Search(config.get("search", {}))

    sizes = ", ".join(
        f"{src}: {sum(e.size for e in engines)} [{','.join(e.type_name for e in engines)}]"
        for src, engines in s.engines_by_source.items()
    )

    print(f"Search: {sizes}", file=sys.stderr)
    print("> ", end="", file=sys.stderr, flush=True)

    for line in sys.stdin:
        query = line.strip()

        if not query:
            print("> ", end="", file=sys.stderr, flush=True)

            continue

        hits = s.search(query, 10)

        for h in hits:
            print(f"{RED}[rrf={h['rank']:.4f}]{RESET}")

            for m in h["members"]:
                src_c = SOURCE_COLORS.get(m["source"], "")
                eng_c = ENGINE_COLORS.get(m["engine"], "")
                paths = ", ".join(f"{src_c}{p}{RESET}" for p in m["paths"])

                print(f"{DIM}[raw={m['raw_score']:.3f}]{RESET} {eng_c}{m['engine']}{RESET} {paths}")
                print(m["data"])

            print(f"{DIM}---{RESET}")

        print("> ", end="", file=sys.stderr, flush=True)
