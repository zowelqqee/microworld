"""Concept graph with spreading activation.

Direct port of the reasoning mechanism in
``worldpgt/cognition/semantic_thought_graph.py``. The production system builds
a small typed graph (task, subject, evidence, gap, pattern, move) and calls
``_activate`` — a bounded max-spread propagation with a fixed damping factor —
to select which cognitive moves light up. We keep that algorithm byte-for-byte
in ``activate`` below; only the node vocabulary changed:

    production node kinds : task / subject / evidence / gap / pattern / move
    poetry node kinds     : concept (imagery word)

Edges are learned co-occurrence associations from the corpus (which images
appear together in a line/stanza), i.e. the poetic analogue of typed relations.
Given seed concepts drawn from a prompt, activation spreads outward and returns
the ranked set of associated images — the reasoning step that decides *what to
write about* before any word is chosen.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ConceptGraph:
    weight: dict[str, float] = field(default_factory=dict)
    edges: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    # epithet[noun] -> Counter-like {adjective: freq}: adjectives observed
    # modifying a noun in the corpus. The imagery analogue of the production
    # graph's typed relations; used by the generator for image phrases.
    epithet: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    # Lazily built adjacency index for neighbors(); scanning all of `edges`
    # per call was fine at hundreds of edges but became the bottleneck once
    # a full-works corpus pushed edge counts into the millions. Not part of
    # to_dict/from_dict — it is derived from `edges`, not learned state.
    _adjacency: dict[str, list[tuple[str, float]]] | None = field(
        default=None, repr=False, compare=False
    )

    def add_edge(self, a: str, b: str, w: float = 1.0) -> None:
        if a == b:
            return
        self.edges[(a, b)] += w
        self.edges[(b, a)] += w
        self._adjacency = None

    def to_dict(self) -> dict:
        return {
            "weight": self.weight,
            "edges": {f"{a}\t{b}": w for (a, b), w in self.edges.items()},
            "epithet": {n: adj for n, adj in self.epithet.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptGraph":
        g = cls()
        g.weight = dict(data.get("weight", {}))
        for key, w in data.get("edges", {}).items():
            a, b = key.split("\t")
            g.edges[(a, b)] = w
        for n, adj in data.get("epithet", {}).items():
            g.epithet[n] = dict(adj)
        return g

    def _ensure_adjacency(self) -> dict[str, list[tuple[str, float]]]:
        if self._adjacency is None:
            adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for (a, b), w in self.edges.items():
                adj[a].append((b, w))
            self._adjacency = adj
        return self._adjacency

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        return self._ensure_adjacency().get(node, [])

    def activate(self, seeds: list[str], *, rounds: int = 3, damping: float = 0.35) -> dict[str, float]:
        """Spreading activation — port of ``semantic_thought_graph._activate``.

        Same structure: seed the activation from node base weights, then for
        ``rounds`` passes push ``source_value * edge_weight * damping`` to each
        target, keeping the max rather than summing. Max-propagation (not sum)
        is what keeps a few hub images from swamping everything, exactly as in
        production.
        """

        activation: dict[str, float] = {n: self.weight.get(n, 0.5) for n in self.weight}
        for s in seeds:
            activation[s] = activation.get(s, 1.0) + 3.0  # inject prompt seeds
        outgoing = self._ensure_adjacency()
        # normalise edge weights per source so damping behaves like production
        for _ in range(rounds):
            nxt = dict(activation)
            for source, source_value in activation.items():
                for target, w in outgoing.get(source, ()):
                    spread = source_value * _norm_weight(w) * damping
                    if spread > nxt.get(target, 0.0):
                        nxt[target] = spread
            activation = nxt
        return activation

    def top_concepts(self, seeds: list[str], k: int = 12) -> list[tuple[str, float]]:
        activation = self.activate(seeds)
        ranked = sorted(activation.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def activate_relevant(
        self, seeds: list[str], *, rounds: int = 2, damping: float = 0.45,
    ) -> dict[str, float]:
        """Activate only the connected field rooted in ``seeds``.

        ``activate`` deliberately keeps global base weights because the poetry
        planner uses them as a broad image prior.  Narrative grounding has a
        different contract: a request about Moscow must not be pulled toward a
        globally frequent person just because that person dominates the corpus.
        This variant starts at the requested nodes and propagates through their
        observed edges only.  It is still the same bounded, weighted graph
        traversal; it merely has no global-frequency back door.
        """

        active = {seed: 4.0 for seed in seeds if seed in self.weight}
        frontier = dict(active)
        outgoing = self._ensure_adjacency()
        for _ in range(rounds):
            nxt: dict[str, float] = {}
            for source, source_value in frontier.items():
                for target, weight in outgoing.get(source, ()):
                    spread = source_value * _norm_weight(weight) * damping
                    if spread <= 0:
                        continue
                    if spread > active.get(target, 0.0):
                        active[target] = spread
                    if spread > nxt.get(target, 0.0):
                        nxt[target] = spread
            frontier = nxt
            if not frontier:
                break
        return active


def _norm_weight(w: float) -> float:
    # Squash raw co-occurrence counts into the (0,1] band the production
    # damping constant was tuned against, so a pair seen 20 times does not
    # spread 20x harder than a pair seen once.
    return 1.0 - 1.0 / (1.0 + w)
