"""serve subcommand: start the proxy server."""

import argparse
import os
import sys

import uvicorn

import proxy
from proxy import CONFIG, ENDPOINTS, app, load_config
from rag import RAG


def cmd_serve(args: argparse.Namespace):
    load_config(args.config)
    if args.debug:
        CONFIG["debug"] = True
    if args.port:
        CONFIG["port"] = args.port

    host = CONFIG.get("host", "127.0.0.1")
    port = CONFIG.get("port", 8082)

    def info(msg): print(msg, file=sys.stderr, flush=True)

    info(f"Proxy starting on {host}:{port}")
    info("Endpoints:")
    for role, ep in ENDPOINTS.items():
        info(f"  {role} [{ep['protocol']}]: {ep['base_url']} -> {ep['model']}")
    if "compress" in ENDPOINTS:
        info(f"Compression: keep={CONFIG['compress_keep']}, min={CONFIG['compress_min']}")

    rag_dirs = CONFIG.get("rag_dirs", [])
    if rag_dirs:
        exts = CONFIG.get("rag_extensions")
        chunk_size = CONFIG.get("rag_chunk_size", 2000)
        dirs = [os.path.expanduser(d) for d in rag_dirs]
        proxy.rag_instance = RAG(dirs, set(exts) if exts else None, chunk_size)
        info(f"RAG: {proxy.rag_instance.n_files} files, {proxy.rag_instance.n_chunks} chunks from {', '.join(dirs)}")

    if CONFIG["debug"]:
        info("Debug: ENABLED (JSONL to stdout, redirect with > debug.jsonl)")
    info("")
    info("Usage:")
    info(f"  ANTHROPIC_BASE_URL=http://{host}:{port} ANTHROPIC_API_KEY=dummy claude")

    uvicorn.run(app, host=host, port=port, log_level="info")
