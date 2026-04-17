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
