"""Build the wiki-pattern knowledge probe benchmark v1.

Reads curated wiki snippets + accepted_auto review items, extracts deterministic
sentence/phrase pattern candidates, and writes a 60-row probe benchmark CSV.

READ-ONLY with respect to all source files, memory, policy, and benchmark outputs.

Usage:
    python3 -m worldpgt.experiments.build_knowledge_probe_benchmark_v1 \\
      --snippets worldpgt/experiments/curated_wiki_snippets_v1.json \\
      --auto-review worldpgt/experiments/knowledge_ingestion_v1_auto_review.json \\
      --patterns-out worldpgt/experiments/wiki_pattern_candidates_v1.json \\
      --probe-out worldpgt/experiments/knowledge_probe_prompts_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

from worldpgt.knowledge.wiki_pattern_extractor import WikiPatternExtractor
from worldpgt.knowledge.wiki_pattern_types import PatternCandidate

# ---------------------------------------------------------------------------
# Probe benchmark: 60 rows — 6 terms × 2 senses × 5 prompts
# Row fields: row_id, term, target_sense, prompt, expected_safe_behavior,
#             knowledge_focus, expected_cue, expected_phrase_type, risk_level
# ---------------------------------------------------------------------------

_PROBE_ROWS: list[dict] = [
    # -----------------------------------------------------------------------
    # bank : financial_institution  (kp-v1-001 .. kp-v1-005)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-001",
        "term": "bank",
        "target_sense": "financial_institution",
        "prompt": "The customer opened a savings account at the bank",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "financial_transaction",
        "expected_cue": "account",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-002",
        "term": "bank",
        "target_sense": "financial_institution",
        "prompt": "She asked the teller for an ATM card at the bank",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "financial_transaction",
        "expected_cue": "teller",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-003",
        "term": "bank",
        "target_sense": "financial_institution",
        "prompt": "He checked the interest rate on his loan at the bank",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "financial_transaction",
        "expected_cue": "interest",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-004",
        "term": "bank",
        "target_sense": "financial_institution",
        "prompt": "The bank branch processed the check and credited the account",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "financial_transaction",
        "expected_cue": "branch",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-005",
        "term": "bank",
        "target_sense": "financial_institution",
        "prompt": "The bank beside the reeds also had a branch office for savings",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "financial_transaction",
        "expected_cue": "savings",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # bank : river_edge  (kp-v1-006 .. kp-v1-010)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-006",
        "term": "bank",
        "target_sense": "river_edge",
        "prompt": "The fishermen cast their lines from the bank of the river",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "natural_geography",
        "expected_cue": "fishermen",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-007",
        "term": "bank",
        "target_sense": "river_edge",
        "prompt": "Tall reeds and grass grew along the bank beside the stream",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "natural_geography",
        "expected_cue": "reed",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-008",
        "term": "bank",
        "target_sense": "river_edge",
        "prompt": "Floodwater rose over the bank and soaked the soil",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "natural_geography",
        "expected_cue": "floodwater",
        "expected_phrase_type": "causal",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-009",
        "term": "bank",
        "target_sense": "river_edge",
        "prompt": "The grassy bank beside the stream dropped into the current below",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "natural_geography",
        "expected_cue": "grass",
        "expected_phrase_type": "location",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-010",
        "term": "bank",
        "target_sense": "river_edge",
        "prompt": "The bank manager noticed reeds growing across the river near the branch",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "natural_geography",
        "expected_cue": "reed",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # bat : animal  (kp-v1-011 .. kp-v1-015)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-011",
        "term": "bat",
        "target_sense": "animal",
        "prompt": "The bat roosted in the rafters of the old barn at dusk",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "roost",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-012",
        "term": "bat",
        "target_sense": "animal",
        "prompt": "Bats use echolocation to hunt insects in the dark",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "echolocation",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-013",
        "term": "bat",
        "target_sense": "animal",
        "prompt": "The bat hung from the ceiling with its membrane wings folded",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "membrane",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-014",
        "term": "bat",
        "target_sense": "animal",
        "prompt": "The flying mammal bat emerged from the attic at night",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "mammal",
        "expected_phrase_type": "definition",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-015",
        "term": "bat",
        "target_sense": "animal",
        "prompt": "The bat flew toward the batter during the game",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "roost",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # bat : sports_equipment  (kp-v1-016 .. kp-v1-020)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-016",
        "term": "bat",
        "target_sense": "sports_equipment",
        "prompt": "The pitcher threw and the batter hit a home run with the bat",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "sports_equipment_use",
        "expected_cue": "pitcher",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-017",
        "term": "bat",
        "target_sense": "sports_equipment",
        "prompt": "The aluminum bat cracked on contact with the ball",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "sports_equipment_use",
        "expected_cue": "aluminum",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-018",
        "term": "bat",
        "target_sense": "sports_equipment",
        "prompt": "He swung the bat at the pitch and drove the ball deep",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "sports_equipment_use",
        "expected_cue": "pitch",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-019",
        "term": "bat",
        "target_sense": "sports_equipment",
        "prompt": "The aluminum bat connected with the ball and the crowd cheered",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "sports_equipment_use",
        "expected_cue": "ball",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-020",
        "term": "bat",
        "target_sense": "sports_equipment",
        "prompt": "The bat dropped from the cave roof as the batter stepped to the plate",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "sports_equipment_use",
        "expected_cue": "pitcher",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # seal : animal  (kp-v1-021 .. kp-v1-025)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-021",
        "term": "seal",
        "target_sense": "animal",
        "prompt": "The seal pup rested on the rocky coast near the water",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "pup",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-022",
        "term": "seal",
        "target_sense": "animal",
        "prompt": "A marine mammal like the seal hunts fish near the shore",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "marine",
        "expected_phrase_type": "definition",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-023",
        "term": "seal",
        "target_sense": "animal",
        "prompt": "The seal barked loudly as the orca approached the coast",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "orca",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-024",
        "term": "seal",
        "target_sense": "animal",
        "prompt": "Seals have strong flippers for swimming and diving in the ocean",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "flipper",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-025",
        "term": "seal",
        "target_sense": "animal",
        "prompt": "The clerk pressed the official seal near the marine exhibit on the coast",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "animal_behavior",
        "expected_cue": "marine",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # seal : closure_stamp  (kp-v1-026 .. kp-v1-030)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-026",
        "term": "seal",
        "target_sense": "closure_stamp",
        "prompt": "The notary pressed the embossed seal onto the certificate",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "document_closure",
        "expected_cue": "notary",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-027",
        "term": "seal",
        "target_sense": "closure_stamp",
        "prompt": "An airtight seal kept the jar closed during transport",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "document_closure",
        "expected_cue": "airtight",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-028",
        "term": "seal",
        "target_sense": "closure_stamp",
        "prompt": "The stamped seal on the official document was authentic",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "document_closure",
        "expected_cue": "stamped",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-029",
        "term": "seal",
        "target_sense": "closure_stamp",
        "prompt": "The certificate required a notary seal and a clerk signature",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "document_closure",
        "expected_cue": "certificate",
        "expected_phrase_type": "relation",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-030",
        "term": "seal",
        "target_sense": "closure_stamp",
        "prompt": "The seal barked near the stamped parcel on the pier",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "document_closure",
        "expected_cue": "stamped",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # crane : bird  (kp-v1-031 .. kp-v1-035)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-031",
        "term": "crane",
        "target_sense": "bird",
        "prompt": "A flock of cranes migrated south over the lake at dawn",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "bird_behavior",
        "expected_cue": "flock",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-032",
        "term": "crane",
        "target_sense": "bird",
        "prompt": "The crane waded through the wetland in search of fish",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "bird_behavior",
        "expected_cue": "wetland",
        "expected_phrase_type": "location",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-033",
        "term": "crane",
        "target_sense": "bird",
        "prompt": "Its long legs helped the crane stand in the shallow reeds",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "bird_behavior",
        "expected_cue": "leg",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-034",
        "term": "crane",
        "target_sense": "bird",
        "prompt": "Cranes are known for their migration in large flocks across wetlands",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "bird_behavior",
        "expected_cue": "migration",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-035",
        "term": "crane",
        "target_sense": "bird",
        "prompt": "The crane at the construction site spread its wing while a flock flew over",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "bird_behavior",
        "expected_cue": "flock",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # crane : machine  (kp-v1-036 .. kp-v1-040)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-036",
        "term": "crane",
        "target_sense": "machine",
        "prompt": "The crane's cable lowered the steel beams to the ground",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "heavy_machinery_operation",
        "expected_cue": "cable",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-037",
        "term": "crane",
        "target_sense": "machine",
        "prompt": "The boom of the tower crane rose above the building site",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "heavy_machinery_operation",
        "expected_cue": "boom",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-038",
        "term": "crane",
        "target_sense": "machine",
        "prompt": "Outriggers kept the crane stable on the uneven construction site",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "heavy_machinery_operation",
        "expected_cue": "outrigger",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-039",
        "term": "crane",
        "target_sense": "machine",
        "prompt": "The operator used the crane to swing the steel beams over the site",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "heavy_machinery_operation",
        "expected_cue": "steel beams",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-040",
        "term": "crane",
        "target_sense": "machine",
        "prompt": "The crane spread its neck near the tower crane on the site",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "heavy_machinery_operation",
        "expected_cue": "tower crane",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # rock : stone  (kp-v1-041 .. kp-v1-045)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-041",
        "term": "rock",
        "target_sense": "stone",
        "prompt": "The geologist studied the mineral layers in the rock formation",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "natural_material",
        "expected_cue": "mineral",
        "expected_phrase_type": "relation",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-042",
        "term": "rock",
        "target_sense": "stone",
        "prompt": "A boulder of solid rock blocked the trail on the hillside",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "natural_material",
        "expected_cue": "hillside",
        "expected_phrase_type": "location",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-043",
        "term": "rock",
        "target_sense": "stone",
        "prompt": "Geology experts noted the rock formation near the mountain trail",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "natural_material",
        "expected_cue": "geology",
        "expected_phrase_type": "relation",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-044",
        "term": "rock",
        "target_sense": "stone",
        "prompt": "The heavy rock lay near the mountain stream at the base of the cliff",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "natural_material",
        "expected_cue": "mineral",
        "expected_phrase_type": "location",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-045",
        "term": "rock",
        "target_sense": "stone",
        "prompt": "The rock band rehearsed near the cliff while climbers scaled the boulder",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "natural_material",
        "expected_cue": "mineral",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # rock : music  (kp-v1-046 .. kp-v1-050)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-046",
        "term": "rock",
        "target_sense": "music",
        "prompt": "The rock band released a new album to sold-out crowds",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "music_performance",
        "expected_cue": "album",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-047",
        "term": "rock",
        "target_sense": "music",
        "prompt": "He played drums on the stage at the rock concert",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "music_performance",
        "expected_cue": "drum",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-048",
        "term": "rock",
        "target_sense": "music",
        "prompt": "Rock is a genre that includes punk and metal subgenres",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "music_performance",
        "expected_cue": "genre",
        "expected_phrase_type": "definition",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-049",
        "term": "rock",
        "target_sense": "music",
        "prompt": "The rock crowd cheered as the band played on the stage",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "music_performance",
        "expected_cue": "album",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-050",
        "term": "rock",
        "target_sense": "music",
        "prompt": "The drum kit sat on the flat rock near the cliff by the stage",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "music_performance",
        "expected_cue": "drum",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # spring : season  (kp-v1-051 .. kp-v1-055)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-051",
        "term": "spring",
        "target_sense": "season",
        "prompt": "Spring flowers bloomed in the garden after the morning rain",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "seasonal_transition",
        "expected_cue": "bloom",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-052",
        "term": "spring",
        "target_sense": "season",
        "prompt": "The spring migration brought birds to the lake every year",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "seasonal_transition",
        "expected_cue": "migration",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-053",
        "term": "spring",
        "target_sense": "season",
        "prompt": "Fresh spring mornings followed the thaw each year",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "seasonal_transition",
        "expected_cue": "morning",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-054",
        "term": "spring",
        "target_sense": "season",
        "prompt": "Spring was the season when flowers bloomed and birds returned",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "seasonal_transition",
        "expected_cue": "flower",
        "expected_phrase_type": "definition",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-055",
        "term": "spring",
        "target_sense": "season",
        "prompt": "The spring in the latch clicked as the spring rain filled the garden",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "seasonal_transition",
        "expected_cue": "bloom",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
    # -----------------------------------------------------------------------
    # spring : coil  (kp-v1-056 .. kp-v1-060)
    # -----------------------------------------------------------------------
    {
        "row_id": "kp-v1-056",
        "term": "spring",
        "target_sense": "coil",
        "prompt": "The coil spring in the door handle under constant tension",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "mechanical_device",
        "expected_cue": "coil",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-057",
        "term": "spring",
        "target_sense": "coil",
        "prompt": "An elastic spring in the mattress provided comfortable support",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "mechanical_device",
        "expected_cue": "mattress",
        "expected_phrase_type": "property",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-058",
        "term": "spring",
        "target_sense": "coil",
        "prompt": "The worn spring in the latch caused the handle to stick",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "mechanical_device",
        "expected_cue": "elastic",
        "expected_phrase_type": "causal",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-059",
        "term": "spring",
        "target_sense": "coil",
        "prompt": "The elastic spring inside the device maintained tension throughout the bounce",
        "expected_safe_behavior": "continue",
        "knowledge_focus": "mechanical_device",
        "expected_cue": "elastic",
        "expected_phrase_type": "action",
        "risk_level": "low",
    },
    {
        "row_id": "kp-v1-060",
        "term": "spring",
        "target_sense": "coil",
        "prompt": "The metal spring latch compressed as the spring flowers bloomed in the warm garden",
        "expected_safe_behavior": "audit",
        "knowledge_focus": "mechanical_device",
        "expected_cue": "coil",
        "expected_phrase_type": "conflict",
        "risk_level": "medium",
    },
]

_PROBE_FIELDS = [
    "row_id", "term", "target_sense", "prompt",
    "expected_safe_behavior", "knowledge_focus",
    "expected_cue", "expected_phrase_type", "risk_level",
]


def _candidates_to_json(candidates: list[PatternCandidate]) -> list[dict]:
    out: list[dict] = []
    for c in candidates:
        d = dataclasses.asdict(c)
        out.append(d)
    return out


def run(
    snippets_path: str,
    auto_review_path: str,
    patterns_out_path: str,
    probe_out_path: str,
) -> dict:
    """Extract patterns and write probe benchmark. Returns a stats dict."""
    with open(snippets_path, encoding="utf-8") as f:
        snippets: list[dict] = json.load(f)

    extractor = WikiPatternExtractor()
    candidates = extractor.extract(snippets)

    decisions: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for c in candidates:
        decisions[c.decision] = decisions.get(c.decision, 0) + 1
        by_type[c.pattern_type] = by_type.get(c.pattern_type, 0) + 1

    pattern_payload = {
        "pattern_candidates": _candidates_to_json(candidates),
        "stats": {
            "total": len(candidates),
            "by_decision": decisions,
            "by_pattern_type": by_type,
        },
    }

    with open(patterns_out_path, "w", encoding="utf-8") as f:
        json.dump(pattern_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    with open(probe_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PROBE_FIELDS)
        writer.writeheader()
        writer.writerows(_PROBE_ROWS)

    return {
        "pattern_candidates": len(candidates),
        "decisions": decisions,
        "by_pattern_type": by_type,
        "probe_rows": len(_PROBE_ROWS),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build wiki-pattern knowledge probe benchmark v1."
    )
    parser.add_argument("--snippets", required=True, help="curated_wiki_snippets_v1.json")
    parser.add_argument("--auto-review", required=True, dest="auto_review",
                        help="knowledge_ingestion_v1_auto_review.json")
    parser.add_argument("--patterns-out", required=True, dest="patterns_out",
                        help="Output wiki_pattern_candidates_v1.json")
    parser.add_argument("--probe-out", required=True, dest="probe_out",
                        help="Output knowledge_probe_prompts_v1.csv")
    args = parser.parse_args(argv)

    stats = run(
        snippets_path=args.snippets,
        auto_review_path=args.auto_review,
        patterns_out_path=args.patterns_out,
        probe_out_path=args.probe_out,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
