"""Reciprocal Rank Fusion over hits connected by shared paths."""

import claudex.con_com as cc


def fuse(raw_hits: list[dict], k: int = 60) -> list[dict]:
    """Merge hits whose path sets overlap (transitively). Sum per-engine 1/(k+pos+1)
    contributions into a single rank; keep the shortest data as the group's snippet."""
    groups = cc.connect_by_shared_elements([h["paths"] for h in raw_hits])
    fused = []

    for indices in groups:
        members = [raw_hits[i] for i in indices]
        rank = sum(1.0 / (k + m["pos"] + 1) for m in members)
        paths: frozenset = frozenset().union(*(m["paths"] for m in members))
        shortest = min(members, key=lambda m: len(m["data"]))

        entry = {**shortest, "paths": sorted(paths), "rank": rank}
        entry.pop("pos", None)

        fused.append(entry)

    fused.sort(key=lambda r: -r["rank"])

    return fused


def test():
    # empty
    assert fuse([]) == []

    # single hit passes through with rank = 1/(k+1)
    out = fuse([{
        "paths": frozenset(["a"]), "data": "x", "source": "fs",
        "engine": "whoosh", "raw_score": 1.0, "pos": 0,
    }])
    assert len(out) == 1
    assert out[0]["paths"] == ["a"]
    assert out[0]["rank"] == 1.0 / 61
    assert "pos" not in out[0]

    # two non-overlapping hits -> two groups, sorted by rank desc
    out = fuse([
        {"paths": frozenset(["a"]), "data": "d1", "source": "fs", "engine": "e", "raw_score": 1.0, "pos": 1},
        {"paths": frozenset(["b"]), "data": "d2", "source": "fs", "engine": "e", "raw_score": 1.0, "pos": 0},
    ])
    assert [h["paths"] for h in out] == [["b"], ["a"]]

    # same path, two engines -> one group with summed rank and shortest data
    out = fuse([
        {"paths": frozenset(["a"]), "data": "long data", "source": "fs", "engine": "whoosh", "raw_score": 2.0, "pos": 0},
        {"paths": frozenset(["a"]), "data": "x",        "source": "fs", "engine": "rag",    "raw_score": 0.5, "pos": 0},
    ])
    assert len(out) == 1
    assert out[0]["paths"] == ["a"]
    assert out[0]["data"] == "x"
    assert out[0]["engine"] == "rag"
    assert out[0]["raw_score"] == 0.5
    assert abs(out[0]["rank"] - 2 * (1.0 / 61)) < 1e-12

    # transitive overlap across three hits via shared paths
    out = fuse([
        {"paths": frozenset(["a", "b"]), "data": "AAAA", "source": "fs", "engine": "e", "raw_score": 1.0, "pos": 0},
        {"paths": frozenset(["b", "c"]), "data": "B",    "source": "fs", "engine": "e", "raw_score": 1.0, "pos": 1},
        {"paths": frozenset(["c", "d"]), "data": "CCCC", "source": "fs", "engine": "e", "raw_score": 1.0, "pos": 2},
        {"paths": frozenset(["z"]),      "data": "Z",    "source": "fs", "engine": "e", "raw_score": 1.0, "pos": 0},
    ])
    assert len(out) == 2
    merged = next(h for h in out if "a" in h["paths"])
    alone = next(h for h in out if h["paths"] == ["z"])
    assert merged["paths"] == ["a", "b", "c", "d"]
    assert merged["data"] == "B"
    expected = 1.0 / 61 + 1.0 / 62 + 1.0 / 63
    assert abs(merged["rank"] - expected) < 1e-12
    assert alone["rank"] == 1.0 / 61

    # tie-break on equal-length data: first member wins via min() stability
    out = fuse([
        {"paths": frozenset(["a"]), "data": "yy", "source": "fs", "engine": "first",  "raw_score": 0, "pos": 0},
        {"paths": frozenset(["a"]), "data": "zz", "source": "fs", "engine": "second", "raw_score": 0, "pos": 1},
    ])
    assert len(out) == 1
    assert out[0]["engine"] == "first"

    # k parameter affects magnitude but not relative order
    big_k = fuse([
        {"paths": frozenset(["a"]), "data": "x", "source": "fs", "engine": "e", "raw_score": 0, "pos": 0},
        {"paths": frozenset(["b"]), "data": "x", "source": "fs", "engine": "e", "raw_score": 0, "pos": 1},
    ], k=1000)
    assert big_k[0]["rank"] == 1.0 / 1001
    assert big_k[1]["rank"] == 1.0 / 1002
