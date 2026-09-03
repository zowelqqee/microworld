"""Legal QA lane — statutory question answering over the explicit graph.

A sibling of ``entity_qa`` / ``cross_page_qa`` / ``multihop_qa``: the codebase
already routes different *question shapes* to different lanes, and a statutory
question is a shape none of the existing lanes parses. The legal QA study
showed why this lane is the missing piece — 45 of 46 failures were the query
layer failing to reach knowledge the graph already held, not the graph lacking
it.

Deterministic. No ML, no network, no overlay writes.
"""
