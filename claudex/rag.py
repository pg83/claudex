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
    def __init__(
        self,
        db_path: str,
        directories: list = None,
        extensions: set = None,
        chunk_size: int = 2000,
        embedder: Callable[[str], list[float]] = default_embedder,
    ):
        self.embedder = embedder
        self.cache: dict[str, list[float]] = {}

        db_path = os.path.expanduser(db_path)
        parent = os.path.dirname(db_path)

        if parent:
            os.makedirs(parent, exist_ok=True)

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

        if directories:
            if isinstance(directories, str):
                directories = [directories]

            extensions = extensions or DEFAULT_EXTENSIONS

            for directory in directories:
                for root, dirs, files in os.walk(directory):
                    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

                    for fname in files:
                        if os.path.splitext(fname)[1] not in extensions:
                            continue

                        fpath = os.path.join(root, fname)
                        rel = os.path.abspath(fpath)

                        try:
                            text = open(fpath, errors="replace").read()
                        except (OSError, UnicodeDecodeError):
                            continue

                        added = self.add(rel, text, chunk_size)
                        print(f"  rag: {rel} (+{added} chunks, {len(text)} chars)", file=sys.stderr, flush=True)

    def add(self, path: str, text: str, chunk_size: int = 2000) -> int:
        added = 0
        dirty = False

        for chunk in split_text(text, chunk_size):
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

    def search(self, query: str, limit: int = 3) -> list[dict]:
        if not self.cache:
            return []

        qvec = self.embedder(query)
        scored = [(cosine(qvec, vec), sha) for sha, vec in self.cache.items()]
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[1:limit + 1]

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
