import os
import sys
import math
import httpx
import pickle
import hashlib
import sqlite3

from typing import Callable

import claudex.rrf as rrf


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


def walk_files(cfg: dict):
    dirs = [os.path.expanduser(d) for d in cfg["dirs"]]
    extensions = set(cfg["extensions"]) if cfg.get("extensions") else DEFAULT_EXTENSIONS

    for directory in dirs:
        for root, subdirs, files in os.walk(directory):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]

            for fname in files:
                if os.path.splitext(fname)[1] not in extensions:
                    continue

                if fname.endswith("_ut.cpp"):
                    continue

                fpath = os.path.join(root, fname)
                rel = os.path.abspath(fpath)

                try:
                    text = open(fpath, errors="replace").read()
                except (OSError, UnicodeDecodeError):
                    continue

                yield rel, text


# ---------------------------------------------------------------------------
# RAG engine (sqlite + embedding cosine)
# ---------------------------------------------------------------------------


class RAG:
    type_name = "rag"

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

        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks (sha TEXT PRIMARY KEY, data TEXT, embedding BLOB)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS paths (sha TEXT, path TEXT, PRIMARY KEY (sha, path))"
        )

        for sha, emb_blob in self.db.execute("SELECT sha, embedding FROM chunks"):
            self.cache[sha] = pickle.loads(emb_blob)

        for path, text in walk_files(cfg):
            added = self.add(path, text)
            print(f"  rag: {path} (+{added} chunks, {len(text)} chars)", file=sys.stderr, flush=True)

    def add(self, path: str, text: str) -> int:
        added = 0
        dirty = False

        for chunk in split_text(text, self.chunk_size):
            if len(chunk.strip()) < 50:
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

    def search(self, query: str) -> list[dict]:
        if not self.cache:
            return []

        qvec = self.embedder(query)
        scored = [(cosine(qvec, vec), sha) for sha, vec in self.cache.items()]
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[:self.max_results]

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

    @property
    def size(self) -> int:
        return len(self.cache)


# ---------------------------------------------------------------------------
# Whoosh engine (BM25 keyword search, in-memory)
# ---------------------------------------------------------------------------


class WhooshEngine:
    type_name = "whoosh"

    def __init__(self, cfg: dict):
        from whoosh.fields import Schema, TEXT, ID
        from whoosh.filedb.filestore import RamStorage

        self.max_results = cfg.get("max_results", 5)
        self.snippet_chars = cfg.get("snippet_chars", 300)

        self._schema = Schema(
            path=ID(stored=True, unique=True),
            body=TEXT(stored=True),
        )
        self._ix = RamStorage().create_index(self._schema)

        writer = self._ix.writer()
        n = 0

        for path, text in walk_files(cfg):
            if not text.strip():
                continue

            writer.update_document(path=path, body=text)
            n += 1
            print(f"  whoosh: {path} ({len(text)} chars)", file=sys.stderr, flush=True)

        writer.commit()
        self._doc_count = n

    def add(self, path: str, text: str) -> int:
        if not text.strip():
            return 0

        writer = self._ix.writer()
        writer.update_document(path=path, body=text)
        writer.commit()
        self._doc_count += 1

        return 1

    def search(self, query: str) -> list[dict]:
        from whoosh.qparser import QueryParser, OrGroup
        from whoosh.highlight import ContextFragmenter, NullFormatter

        with self._ix.searcher() as s:
            q = QueryParser("body", self._schema, group=OrGroup.factory(0.9)).parse(query)
            hits = s.search(q, limit=self.max_results)
            hits.fragmenter = ContextFragmenter(maxchars=self.snippet_chars, surround=self.snippet_chars // 4)
            hits.formatter = NullFormatter()

            return [
                {
                    "paths": [h["path"]],
                    "data": h.highlights("body") or h["body"][:self.snippet_chars],
                    "rank": h.score,
                }
                for h in hits
            ]

    @property
    def size(self) -> int:
        return self._doc_count


# ---------------------------------------------------------------------------
# Search dispatcher
# ---------------------------------------------------------------------------


ENGINES = {
    "rag": RAG,
    "whoosh": WhooshEngine,
}


def make_engine(cfg: dict):
    t = cfg.get("type")
    cls = ENGINES.get(t)

    if cls is None:
        raise ValueError(f"unknown search engine type: {t!r}")

    return cls(cfg)


class Search:
    def __init__(self, cfg: dict):
        self.engines_by_source: dict[str, list] = {}

        for source, engines_cfg in cfg.items():
            self.engines_by_source[source] = [make_engine(e) for e in engines_cfg]

    def add(self, path: str, text: str):
        source = "conversation" if path.startswith("conversation/") else "fs"

        for engine in self.engines_by_source.get(source, []):
            engine.add(path, text)

    def search(self, query: str, limit: int = None, exclude_paths: list[str] = None) -> list[dict]:
        excluded = set(exclude_paths or [])
        raw_hits = []

        for source, engines in self.engines_by_source.items():
            for engine in engines:
                for pos, h in enumerate(engine.search(query)):
                    if excluded and any(p in excluded for p in h["paths"]):
                        continue

                    raw_hits.append({
                        "paths": frozenset(h["paths"]),
                        "data": h["data"],
                        "source": source,
                        "engine": engine.type_name,
                        "raw_score": h["rank"],
                        "pos": pos,
                    })

        fused = rrf.fuse(raw_hits)

        if limit is not None:
            fused = fused[:limit]

        return fused


def test():
    import io
    import tempfile
    import contextlib

    with tempfile.TemporaryDirectory() as d:
        alpha = os.path.join(d, "alpha.txt")
        beta = os.path.join(d, "beta.txt")

        with open(alpha, "w") as f:
            f.write("the quick brown fox jumps over the lazy dog")

        with open(beta, "w") as f:
            f.write("another fox appears at dawn beneath silver moons")

        cfg = {"dirs": [d], "max_results": 5, "snippet_chars": 100, "extensions": [".txt"]}

        with contextlib.redirect_stderr(io.StringIO()):
            engine = WhooshEngine(cfg)

        assert engine.size == 2

        hits = engine.search("fox")
        paths = {h["paths"][0] for h in hits}
        assert alpha in paths
        assert beta in paths

        for h in hits:
            assert isinstance(h["paths"], list) and len(h["paths"]) == 1
            assert isinstance(h["data"], str) and h["data"]
            assert isinstance(h["rank"], float)

        hits = engine.search("quick brown")
        assert [h["paths"][0] for h in hits][0] == alpha

        hits = engine.search("dawn")
        assert [h["paths"][0] for h in hits] == [beta]

        assert engine.search("zzzzznotfound") == []

        # add() indexes new docs
        gamma = "/synthetic/gamma.txt"

        with contextlib.redirect_stderr(io.StringIO()):
            engine.add(gamma, "a unique sentinel token: kumquat")

        assert engine.size == 3
        hits = engine.search("kumquat")
        assert [h["paths"][0] for h in hits] == [gamma]
