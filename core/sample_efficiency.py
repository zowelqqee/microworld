"""
Sample-efficiency benchmark.

Question:
    Do consolidation / concept formation / structural similarity let the model
    predict lifecycle-closing edges from FEWER observations than majority-vote?

Setup:
    A deterministic synthetic world with several lifecycle families, each with
    its own *shared* connector node:

        seed fruits     : seed_tree_i  -> seed_fruit_i -> seed      -> seed_tree_i
        pit fruits      : pit_tree_i   -> pit_fruit_i  -> pit       -> pit_tree_i
        egg animals     : animal_i     -> egg_i        -> offspring -> animal_i
        business systems: business_i   -> product_i    -> revenue   -> business_i

    The connector is the family signature (seed / pit / offspring / revenue).
    From each family we hold out instance 0: we observe its produces_result and
    contains edges but HIDE the lifecycle-closing grows_into edge, which becomes
    the prediction target. The held-out set is identical across all modes.

    A train *budget* controls how many of the remaining (pool) instances are
    observed as complete cycles. The pool is ordered family-by-family, so as the
    budget grows, each family's first complete example appears in turn — letting
    us watch how quickly each inference mode generalizes.

Why majority is sample-inefficient here:
    The world has four equally-valid connectors. Majority vote can only bet on
    the single globally-dominant one, so it can never recover more than one
    family's held-out edge — no matter how much data it sees. Structural
    similarity and concept membership reason per-family, so a single example of
    a family is enough to recover that family's held-out edge.
"""
from dataclasses import dataclass
from .world import World
from .relations import Relation
from .evaluation import PredictionEvaluator, EvaluationResult


# (family, tree_template, fruit_template, shared_connector)
FAMILY_SPECS: list[tuple[str, str, str, str]] = [
    ("seed",     "seed_tree_{i}", "seed_fruit_{i}", "seed"),
    ("pit",      "pit_tree_{i}",  "pit_fruit_{i}",  "pit"),
    ("egg",      "animal_{i}",    "egg_{i}",        "offspring"),
    ("business", "business_{i}",  "product_{i}",    "revenue"),
]

MODES: list[str] = [
    "majority_only",
    "structural_similarity",
    "concept_only",
    "hybrid_structural",
]

DEFAULT_BUDGETS: list[float] = [0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass(frozen=True)
class Instance:
    family: str
    index: int
    tree: str
    fruit: str
    connector: str

    def edges(self) -> list[tuple[str, str]]:
        """(verb, object) observations that build this lifecycle, in order."""
        return [
            ("производит", self.fruit),   # tree -> fruit       (produces_result)
            ("содержит",   self.connector),  # fruit -> connector (contains)
            # closing edge handled separately so it can be hidden
        ]


def generate_instances(n_per_family: int) -> list[Instance]:
    """Deterministically generate all instances for every family."""
    instances: list[Instance] = []
    for family, tree_t, fruit_t, connector in FAMILY_SPECS:
        for i in range(n_per_family):
            instances.append(Instance(
                family=family,
                index=i,
                tree=tree_t.format(i=i),
                fruit=fruit_t.format(i=i),
                connector=connector,
            ))
    return instances


class SampleEfficiencyExperiment:
    """Builds budget-limited worlds and benchmarks inference modes on them."""

    def __init__(self, n_per_family: int = 6, held_out_index: int = 0) -> None:
        if n_per_family < 2:
            raise ValueError("need at least 2 instances per family (1 held-out + 1 pool)")
        self.n_per_family = n_per_family
        self.held_out_index = held_out_index
        self.instances = generate_instances(n_per_family)

        # Held-out: one instance per family (deterministic).
        self.held_out: list[Instance] = [
            inst for inst in self.instances if inst.index == held_out_index
        ]
        # Pool: everything else, ordered family-by-family (deterministic).
        self.pool: list[Instance] = [
            inst for inst in self.instances if inst.index != held_out_index
        ]

    # ------------------------------------------------------------------
    # Deterministic budget split
    # ------------------------------------------------------------------

    def budget_pool(self, budget: float) -> list[Instance]:
        """Return the prefix of the pool observed at this budget (deterministic)."""
        if not 0.0 <= budget <= 1.0:
            raise ValueError("budget must be in [0, 1]")
        n_train = round(budget * len(self.pool))
        return self.pool[:n_train]

    def hidden_relations(self) -> list[Relation]:
        """The fixed held-out lifecycle-closing edges (same for all modes)."""
        return [
            Relation(inst.connector, "grows_into", inst.tree)
            for inst in self.held_out
        ]

    # ------------------------------------------------------------------
    # World construction — a FRESH world every call
    # ------------------------------------------------------------------

    def build_world(self, budget: float) -> World:
        """
        Build a fresh world: complete cycles for the budgeted pool plus the
        held-out anchors (produces_result + contains observed, closing hidden).
        """
        w = World()
        # Training pool: full lifecycles.
        for inst in self.budget_pool(budget):
            w.observe(f"{inst.tree} производит {inst.fruit}")
            w.observe(f"{inst.fruit} содержит {inst.connector}")
            w.observe(f"{inst.connector} вырастает в {inst.tree}")
        # Held-out anchors: observe everything EXCEPT the closing grows_into.
        for inst in self.held_out:
            w.observe(f"{inst.tree} производит {inst.fruit}")
            w.observe(f"{inst.fruit} содержит {inst.connector}")
        return w

    # ------------------------------------------------------------------
    # Mode evaluation
    # ------------------------------------------------------------------

    def run_mode(self, budget: float, mode: str) -> EvaluationResult:
        """Fresh world, load only this mode's machinery, evaluate. No oracle."""
        w = self.build_world(budget)
        use_sim = use_con = False

        if mode == "majority_only":
            pass
        elif mode == "structural_similarity":
            w.add_structural_similarities(min_score=0.5)
            use_sim = True
        elif mode == "concept_only":
            w.discover_concepts()
            use_con = True
        elif mode == "hybrid_structural":
            w.add_structural_similarities(min_score=0.5)
            w.discover_concepts()
            use_sim = True
            use_con = True
        else:
            raise ValueError(f"unknown mode: {mode}")

        evaluator = PredictionEvaluator()
        return evaluator.evaluate_predictions(
            w, self.hidden_relations(),
            use_similarity=use_sim, use_concepts=use_con,
        )

    def run(
        self,
        budgets: list[float] | None = None,
        modes: list[str] | None = None,
    ) -> list[dict]:
        """Run every (budget, mode) cell and return table rows."""
        budgets = budgets if budgets is not None else DEFAULT_BUDGETS
        modes = modes if modes is not None else MODES
        rows: list[dict] = []
        for budget in budgets:
            for mode in modes:
                r = self.run_mode(budget, mode)
                rows.append({
                    "budget": budget,
                    "mode": mode,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "tp": r.true_positives,
                    "fp": r.false_positives,
                    "fn": r.false_negatives,
                })
        return rows
