"""poemcore — minimal reusable core of the MicroWorld architecture, with the
knowledge source replaced by a Russian-poetry corpus.

Layer map (mirrors the production runtime, QA logic removed):

    ingest.py         knowledge ingestion  (poetry replaces wiki/Reddit)
    concept_graph.py  reasoning: spreading activation over a typed graph
    planner.py        reasoning: move selection into an explicit PoemPlan
    line_plan.py      reasoning: per-line semantic intent (ported SpeechPlan)
    phrase_model.py   language: frequency phrase graph + seeded traversal (order-2)
    discourse.py      language: discourse state + salience selection (ported
                      dialogue state/salience) — cross-line continuity
    generator.py      language: render the plan into metered, rhymed lines
    novelty.py        the support gate, inverted (block memorisation, not
                      unsupported combination)
    engine.py         orchestration: prompt → plan → render → gate

Transferred QA mechanisms, in order: spreading activation, phrase-fragment
context (order-2), discourse salience selection, line-level semantic intent,
intent-seeded generation (the intent steers growth via a must_include hook on
the phrase-model walk, not just candidate ranking).
"""
