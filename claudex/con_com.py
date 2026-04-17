"""Connected components via path-compression union-find."""


class ConnectedComponents:
    """Disjoint-set over N items with path-compression union-find."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)

        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = {}

        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)

        return list(out.values())


def connect_by_shared_elements(sets: list[frozenset]) -> list[list[int]]:
    """Group indices whose sets share any element (transitively)."""
    cc = ConnectedComponents(len(sets))
    by_elem: dict = {}

    for i, s in enumerate(sets):
        for elem in s:
            by_elem.setdefault(elem, []).append(i)

    for indices in by_elem.values():
        first = indices[0]

        for other in indices[1:]:
            cc.union(first, other)

    return cc.groups()


def _norm(groups: list[list[int]]) -> list[list[int]]:
    return sorted(sorted(g) for g in groups)


def test():
    # DSU: empty
    assert ConnectedComponents(0).groups() == []

    # DSU: single element
    assert ConnectedComponents(1).groups() == [[0]]

    # DSU: fully disconnected
    assert _norm(ConnectedComponents(3).groups()) == [[0], [1], [2]]

    # DSU: direct union
    cc = ConnectedComponents(3)
    cc.union(0, 1)
    assert _norm(cc.groups()) == [[0, 1], [2]]

    # DSU: transitive union
    cc = ConnectedComponents(4)
    cc.union(0, 1)
    cc.union(2, 3)
    cc.union(1, 2)
    assert _norm(cc.groups()) == [[0, 1, 2, 3]]

    # DSU: chain collapses to one root
    cc = ConnectedComponents(5)

    for i in range(4):
        cc.union(i, i + 1)

    root = cc.find(0)

    for i in range(5):
        assert cc.find(i) == root

    # shared-elements: empty input
    assert connect_by_shared_elements([]) == []

    # shared-elements: no overlap
    got = connect_by_shared_elements([
        frozenset(["a"]), frozenset(["b"]), frozenset(["c"]),
    ])
    assert _norm(got) == [[0], [1], [2]]

    # shared-elements: direct overlap
    got = connect_by_shared_elements([
        frozenset(["a", "b"]),
        frozenset(["b", "c"]),
        frozenset(["x"]),
    ])
    assert _norm(got) == [[0, 1], [2]]

    # shared-elements: transitive (0-1 via "b", 1-2 via "c")
    got = connect_by_shared_elements([
        frozenset(["a", "b"]),
        frozenset(["b", "c"]),
        frozenset(["c", "d"]),
        frozenset(["z"]),
    ])
    assert _norm(got) == [[0, 1, 2], [3]]

    # shared-elements: same set twice collapses
    got = connect_by_shared_elements([frozenset(["a"]), frozenset(["a"])])
    assert _norm(got) == [[0, 1]]
