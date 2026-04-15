"""Minimal FTS5-based RAG: index a directory, search by text."""

import os
import sqlite3
import sys

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".tox", ".mypy_cache"}
DEFAULT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".sh", ".go", ".js", ".ts", ".rs", ".c", ".h", ".cpp", ".java",
    ".cfg", ".ini", ".env", ".html", ".css", ".sql",
}


class RAG:
    def __init__(self, directories, extensions: set = None, chunk_size: int = 2000):
        if isinstance(directories, str):
            directories = [directories]
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE chunks USING fts5(path, idx, content)")
        extensions = extensions or DEFAULT_EXTENSIONS
        n_files = 0
        n_chunks = 0
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
                    n_files += 1
                    file_chunks = _split(text, chunk_size)
                    for i, chunk in enumerate(file_chunks):
                        self.db.execute("INSERT INTO chunks VALUES (?, ?, ?)", (rel, str(i), chunk))
                        n_chunks += 1
                    print(f"  rag: {rel} ({len(file_chunks)} chunks, {len(text)} chars)", file=sys.stderr, flush=True)
        self.db.commit()
        self.n_files = n_files
        self.n_chunks = n_chunks

    def add(self, path: str, text: str, chunk_size: int = 2000):
        """Add text to the index at runtime."""
        for i, chunk in enumerate(_split(text, chunk_size)):
            self.db.execute("INSERT INTO chunks VALUES (?, ?, ?)", (path, str(i), chunk))
        self.db.commit()

    def search(self, query: str, limit: int = 3) -> list[dict]:
        """Search indexed chunks. Returns [{"path", "idx", "content", "rank"}]."""
        words = _tokenize(query)
        if not words:
            return []
        fts_query = " OR ".join(words)
        rows = self.db.execute(
            "SELECT path, idx, content, rank FROM chunks WHERE content MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        return [{"path": r[0], "idx": int(r[1]), "content": r[2], "rank": r[3]} for r in rows]


def _split(text: str, size: int) -> list[str]:
    """Split text into chunks of ~size chars, breaking at line boundaries."""
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


def _tokenize(text: str) -> list[str]:
    """Extract words suitable for FTS5 MATCH query."""
    import re
    words = re.findall(r"[a-zA-Z_]\w{2,}", text)
    seen = set()
    result = []
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            result.append(wl)
    return result[:20]
