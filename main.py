"""Entry point: interactive REPL for worldmvp."""
import sys
from core import World, AbstractionMiner

_OBSERVE_HELP = """\
worldmvp — вводи наблюдения (пустая строка → режим запросов)

  яблоня дала яблоко        →  produces_result
  яблоко содержит семя      →  contains
  семя вырастает в яблоню   →  grows_into
  дождь вызвал наводнение   →  causes_effect
  удобрение помогает росту  →  enables
  засуха уничтожает урожай  →  prevents
"""

_QUERY_HELP = """\
Команды:
  relations             — все связи
  predict               — предсказания пропущенных связей
  consolidate           — кластеры, циклы, цепочки
  explain <A> <B>       — как A ведёт к B
  add                   — добавить ещё наблюдения
  help                  — эта справка
  quit                  — выход
"""

_DEMO_OBSERVATIONS = [
    "яблоня дала яблоко",
    "яблоко содержит семя",
    "семя вырастает в яблоню",
    "груша дала груша_плод",
    "груша_плод содержит семя",
    "апельсиновое_дерево производит апельсин",
    "апельсин содержит семя",
    "семя вырастает в апельсиновое_дерево",
]


# ── output helpers ──────────────────────────────────────────────────────────

def _show_relations(world: World) -> None:
    rels = world.get_relations()
    if not rels:
        print("  (нет связей)")
        return
    for r in rels:
        ev = f"  [{'; '.join(r.evidence)}]" if r.evidence else ""
        print(f"  {r.source:20s} --{r.relation_type}--> {r.target}{ev}")


def _show_predictions(world: World) -> None:
    preds = world.predict_missing_links()
    if not preds:
        print("  No predictions found")
        return
    for p in preds:
        print(f"  {p.source:20s} --{p.relation_type}--> {p.target:20s}  conf={p.confidence:.2f}")
        print(f"    {p.reason}")


def _show_consolidation(world: World) -> None:
    patterns = world.consolidate()
    if not patterns:
        print("  (нет паттернов)")
        return
    clusters = [p for p in patterns if p.kind == "relation_cluster"]
    cycles   = [p for p in patterns if p.kind == "cycle"]
    chains   = [p for p in patterns if p.kind == "causal_chain"]
    for c in clusters:
        print(f"  cluster [{c.name}] × {len(c.members)}  conf={c.confidence:.2f}")
        for m in c.members:
            print(f"    {m}")
    for cy in cycles:
        print(f"  cycle:  {cy.name.removeprefix('cycle: ')}  conf={cy.confidence:.2f}")
    for ch in chains:
        print(f"  chain:  {ch.name.removeprefix('chain: ')}  conf={ch.confidence:.2f}")


# ── ingestion phase ─────────────────────────────────────────────────────────

def _ingest_loop(world: World) -> None:
    while True:
        try:
            line = input("obs> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise
        if not line:
            return
        event = world.observe(line)
        if event:
            rel = world.get_relations()[-1]
            print(f"  + {event.subject} --{rel.relation_type}--> {event.object}")
        else:
            print("  (не распознано — нужна форма: '<субъект> <глагол> <объект>')")


# ── query phase ─────────────────────────────────────────────────────────────

def _query_loop(world: World) -> None:
    print(_QUERY_HELP)
    while True:
        try:
            raw = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw in ("quit", "exit", "q"):
            break
        cmd, *args = raw.split()
        cmd = cmd.lower()

        if cmd == "relations":
            _show_relations(world)

        elif cmd == "predict":
            _show_predictions(world)

        elif cmd == "consolidate":
            _show_consolidation(world)

        elif cmd == "explain":
            if len(args) < 2:
                print("  использование: explain <источник> <цель>")
            else:
                src, tgt = args[0], args[1]
                known = {o.name for o in world.get_objects()}
                if src not in known:
                    print(f"  неизвестный объект: {src!r} — сначала введи наблюдения с ним")
                elif tgt not in known:
                    print(f"  неизвестный объект: {tgt!r} — сначала введи наблюдения с ним")
                else:
                    chains = world.explain_chain(src, tgt)
                    if not chains or chains == [f"No causal path found from {src!r} to {tgt!r}"]:
                        print(f"  нет пути от {src!r} к {tgt!r}")
                    else:
                        for ch in chains:
                            print(f"  {ch}")

        elif cmd == "add":
            print("  добавляй наблюдения (пустая строка → обратно в запросы)")
            _ingest_loop(world)
            print(_QUERY_HELP)

        elif cmd == "help":
            print(_QUERY_HELP)

        else:
            print(f"  неизвестная команда {cmd!r} — введи 'help'")


# ── entry points ─────────────────────────────────────────────────────────────

def run_demo() -> None:
    print("=== DEMO MODE ===")
    world = World()
    for text in _DEMO_OBSERVATIONS:
        event = world.observe(text)
        if event:
            rel = world.get_relations()[-1]
            print(f"  + {event.subject:22s} --{rel.relation_type}--> {event.object}")
    print()
    _query_loop(world)


def run_interactive() -> None:
    world = World()
    print(_OBSERVE_HELP)
    try:
        _ingest_loop(world)
        _query_loop(world)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()
