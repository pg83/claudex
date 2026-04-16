import os
import sys
import math
import httpx
import pickle
import hashlib
import sqlite3

from typing import Callable


SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".tox", ".mypy_cache"}
DEFAULT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".sh", ".go", ".js", ".ts", ".rs", ".c", ".h", ".cpp", ".java",
    ".cfg", ".ini", ".env", ".html", ".css", ".sql",
}


OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "nomic-embed-text"


def make_ollama_embedder(url: str = OLLAMA_URL, model: str = OLLAMA_MODEL) -> Callable[[str], list[float]]:
    client = httpx.Client(timeout=60.0)

    def embed(text: str) -> list[float]:
        resp = client.post(url, json={"model": model, "input": text})
        resp.raise_for_status()

        return resp.json()["embeddings"][0]

    return embed


default_embedder = make_ollama_embedder()


def split_text(text: str, size: int) -> list[str]:
    chunks = []
    lines = text.split("\n")
    buf = []
    buf_len = 0

    for line in lines:
        if buf_len + len(line) + 1 > size and buf:
            chunks.append("\n".join(buf))
            buf = []
            buf_len = 0

        buf.append(line)
        buf_len += len(line) + 1

    if buf:
        chunks.append("\n".join(buf))

    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0

    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y

    if na == 0.0 or nb == 0.0:
        return 0.0

    return dot / (math.sqrt(na) * math.sqrt(nb))


class RAG:
    def __init__(self, cfg: dict, embedder: Callable[[str], list[float]] = None):
        self.chunk_size = cfg.get("chunk_size", 2000)
        self.max_results = cfg.get("max_results", 3)

        if embedder is None:
            url = cfg.get("embed_url") or OLLAMA_URL
            model = cfg.get("embed_model") or OLLAMA_MODEL
            embedder = make_ollama_embedder(url, model)

        self.embedder = embedder
        self.cache: dict[str, list[float]] = {}

        db_path = os.path.expanduser(cfg.get("db", "~/.cache/claudex/rag.db"))
        parent = os.path.dirname(db_path)

        if parent:
            os.makedirs(parent, exist_ok=True)

        self.db_path = db_path
        self.db = sqlite3.connect(db_path)

        cols = [r[1] for r in self.db.execute("PRAGMA table_info(chunks)")]

        if cols and "chunk" in cols and "data" not in cols:
            self.db.execute("DROP TABLE chunks")

        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks (sha TEXT PRIMARY KEY, data TEXT, embedding BLOB)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS paths (sha TEXT, path TEXT, PRIMARY KEY (sha, path))"
        )

        for sha, emb_blob in self.db.execute("SELECT sha, embedding FROM chunks"):
            self.cache[sha] = pickle.loads(emb_blob)

        dirs = [os.path.expanduser(d) for d in cfg.get("dirs", [])]
        extensions = set(cfg["extensions"]) if cfg.get("extensions") else DEFAULT_EXTENSIONS

        for directory in dirs:
            for root, subdirs, files in os.walk(directory):
                subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]

                for fname in files:
                    if os.path.splitext(fname)[1] not in extensions:
                        continue

                    fpath = os.path.join(root, fname)
                    rel = os.path.abspath(fpath)

                    try:
                        text = open(fpath, errors="replace").read()
                    except (OSError, UnicodeDecodeError):
                        continue

                    added = self.add(rel, text)
                    print(f"  rag: {rel} (+{added} chunks, {len(text)} chars)", file=sys.stderr, flush=True)

    def add(self, path: str, text: str) -> int:
        added = 0
        dirty = False

        for chunk in split_text(text, self.chunk_size):
            if not chunk.strip():
                continue

            sha = hashlib.sha256(chunk.encode()).hexdigest()

            if sha not in self.cache:
                vec = self.embedder(chunk)
                self.db.execute(
                    "INSERT INTO chunks (sha, data, embedding) VALUES (?, ?, ?)",
                    (sha, chunk, pickle.dumps(vec)),
                )
                self.cache[sha] = vec
                added += 1
                dirty = True

            cur = self.db.execute(
                "INSERT OR IGNORE INTO paths (sha, path) VALUES (?, ?)",
                (sha, path),
            )

            if cur.rowcount > 0:
                dirty = True

        if dirty:
            self.db.commit()

        return added

    def search(self, query: str, limit: int = None) -> list[dict]:
        if not self.cache:
            return []

        if limit is None:
            limit = self.max_results

        qvec = self.embedder(query)
        scored = [(cosine(qvec, vec), sha) for sha, vec in self.cache.items()]
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[:limit]

        if not top:
            return []

        shas = [sha for _, sha in top]
        placeholders = ",".join("?" * len(shas))
        data_by_sha = {
            sha: data
            for sha, data in self.db.execute(
                f"SELECT sha, data FROM chunks WHERE sha IN ({placeholders})",
                shas,
            )
        }

        paths_by_sha: dict[str, list[str]] = {}

        for sha, path in self.db.execute(
            f"SELECT sha, path FROM paths WHERE sha IN ({placeholders})",
            shas,
        ):
            paths_by_sha.setdefault(sha, []).append(path)

        return [
            {"paths": paths_by_sha.get(sha, []), "data": data_by_sha.get(sha, ""), "rank": sim}
            for sim, sha in top
            if sha in data_by_sha
        ]


class MultiRAG:
    def __init__(self, cfg: dict):
        parent = {k: v for k, v in cfg.items() if k not in ("files", "conversations")}
        files_cfg = {**parent, **cfg.get("files", {})}
        conv_cfg = {**parent, **cfg.get("conversations", {})}

        self.files = RAG(files_cfg)
        self.conversations = RAG(conv_cfg)

    def add(self, path: str, text: str) -> int:
        if path.startswith("conversation/"):
            return self.conversations.add(path, text)

        return self.files.add(path, text)

    def search(self, query: str, limit: int = None) -> list[dict]:
        f_limit = (limit + 1) if limit else (self.files.max_results + 1)
        c_limit = (limit + 1) if limit else (self.conversations.max_results + 1)

        f_hits = [{**h, "source": "files"} for h in self.files.search(query, f_limit)]
        c_hits = [{**h, "source": "conversations"} for h in self.conversations.search(query, c_limit)]

        merged = sorted(f_hits + c_hits, key=lambda r: -r["rank"])

        if merged:
            merged = merged[1:]

        if limit is not None:
            merged = merged[:limit]

        return merged

    @property
    def cache(self) -> dict:
        return {**self.files.cache, **self.conversations.cache}
