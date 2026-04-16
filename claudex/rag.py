import os
import sys
import json
import math
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


def default_embedder(text: str) -> list[float]:
    return [1.0]


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
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks (sha TEXT PRIMARY KEY, chunk TEXT, embedding BLOB)"
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

        for chunk in split_text(text, chunk_size):
            doc = {"path": path, "data": chunk}
            chunk_json = json.dumps(doc, ensure_ascii=False, sort_keys=True)
            sha = hashlib.sha256(chunk_json.encode()).hexdigest()

            if sha in self.cache:
                continue

            vec = self.embedder(chunk)
            self.db.execute(
                "INSERT INTO chunks (sha, chunk, embedding) VALUES (?, ?, ?)",
                (sha, chunk_json, pickle.dumps(vec)),
            )
            self.cache[sha] = vec
            added += 1

        if added:
            self.db.commit()

        return added

    def search(self, query: str, limit: int = 3) -> list[dict]:
        if not self.cache:
            return []

        qvec = self.embedder(query)
        scored = [(cosine(qvec, vec), sha) for sha, vec in self.cache.items()]
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[:limit]

        if not top:
            return []

        shas = [sha for _, sha in top]
        placeholders = ",".join("?" * len(shas))
        rows = self.db.execute(
            f"SELECT sha, chunk FROM chunks WHERE sha IN ({placeholders})",
            shas,
        ).fetchall()
        by_sha = {sha: json.loads(chunk_json) for sha, chunk_json in rows}

        return [
            {"path": by_sha[sha]["path"], "data": by_sha[sha]["data"], "rank": sim}
            for sim, sha in top
            if sha in by_sha
        ]
