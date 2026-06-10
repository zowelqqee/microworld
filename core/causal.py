from dataclasses import dataclass, field
from .relations import Relation

# How each relation type propagates when the source is removed.
# "affected" = downstream object is impacted but may survive.
# "lost"     = downstream object is directly gone / unreachable.
REMOVAL_IMPACT: dict[str, str] = {
    "produces_result": "lost",       # source gone → result no longer produced
    "causes_effect": "lost",         # cause gone → effect no longer triggered
    "contains": "lost",              # container gone → contained part lost
    "grows_into": "affected",        # seed gone → grown form can't emerge
    "requires": "affected",          # dependency gone → dependent is impaired
    "enables": "affected",           # enabler gone → facilitated thing weakened
    "prevents": "affected",          # preventer gone → blocked thing may emerge
    "reduces": "affected",           # reducer gone → thing may grow
}


@dataclass
class CausalStep:
    object_name: str
    reason: str          # human-readable explanation of why this step is reached

    def __repr__(self) -> str:
        return f"CausalStep({self.object_name!r}: {self.reason})"


@dataclass
class CausalChain:
    steps: list[CausalStep] = field(default_factory=list)

    def __repr__(self) -> str:
        return " → ".join(s.object_name for s in self.steps)


class CausalReasoner:
    """Traverses the relation graph to trace causes and simulate removals."""

    def __init__(self, relations: list[Relation]) -> None:
        self._relations = relations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace_forward(self, start: str, max_depth: int = 3) -> list[CausalChain]:
        """Return all causal chains reachable from *start* up to *max_depth*."""
        results: list[CausalChain] = []
        self._dfs_forward(
            current=start,
            chain=[CausalStep(start, "start")],
            visited={start},
            depth=0,
            max_depth=max_depth,
            results=results,
        )
        return results

    def what_if_removed(self, object_name: str) -> list[str]:
        """
        Simulate removing *object_name* from the world.
        Returns a list of human-readable impact lines, ordered by discovery.
        """
        lines: list[str] = [f"{object_name} removed"]
        affected: set[str] = {object_name}
        queue = [object_name]

        while queue:
            current = queue.pop(0)
            for rel in self._relations:
                if rel.source != current:
                    continue
                if rel.target in affected:
                    continue
                impact = REMOVAL_IMPACT.get(rel.relation_type, "affected")
                lines.append(
                    f"{rel.target} {impact} "
                    f"because {current} --{rel.relation_type}--> {rel.target}"
                )
                affected.add(rel.target)
                queue.append(rel.target)

        return lines

    def explain_chain(self, source: str, target: str) -> list[str]:
        """
        Find all paths from *source* to *target* in the relation graph.
        Returns human-readable path descriptions.
        """
        paths: list[list[tuple[str, str, str]]] = []  # list of (src, rel, tgt) triples
        self._dfs_path(
            current=source,
            target=target,
            path=[],
            visited={source},
            paths=paths,
        )
        if not paths:
            return [f"No causal path found from {source!r} to {target!r}"]

        lines = []
        for path in paths:
            steps = " → ".join(
                f"{s} --{r}--> {t}" for s, r, t in path
            )
            lines.append(steps)
        return lines

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dfs_forward(
        self,
        current: str,
        chain: list[CausalStep],
        visited: set[str],
        depth: int,
        max_depth: int,
        results: list[CausalChain],
    ) -> None:
        outgoing = [r for r in self._relations if r.source == current]
        if not outgoing or depth >= max_depth:
            if len(chain) > 1:
                results.append(CausalChain(list(chain)))
            return
        for rel in outgoing:
            if rel.target in visited:
                continue
            step = CausalStep(
                rel.target,
                f"{current} --{rel.relation_type}--> {rel.target}",
            )
            self._dfs_forward(
                current=rel.target,
                chain=chain + [step],
                visited=visited | {rel.target},
                depth=depth + 1,
                max_depth=max_depth,
                results=results,
            )

    def _dfs_path(
        self,
        current: str,
        target: str,
        path: list[tuple[str, str, str]],
        visited: set[str],
        paths: list[list[tuple[str, str, str]]],
        max_depth: int = 6,
    ) -> None:
        if len(path) >= max_depth:
            return
        for rel in self._relations:
            if rel.source != current:
                continue
            new_path = path + [(rel.source, rel.relation_type, rel.target)]
            # Check target BEFORE visited so we can detect cycles back to the start.
            if rel.target == target:
                paths.append(new_path)
                continue
            if rel.target in visited:
                continue
            self._dfs_path(
                current=rel.target,
                target=target,
                path=new_path,
                visited=visited | {rel.target},
                paths=paths,
                max_depth=max_depth,
            )
