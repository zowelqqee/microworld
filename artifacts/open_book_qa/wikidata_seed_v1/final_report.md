# Wikidata structured-property seed — final report

## Resolution improvement

The original exact resolver covered 41/331 original subjects. The alias/P31 pass produced 9 automatic candidates; manual review rejected 3 contextually wrong aliases/names and confirmed 6, yielding 47/331 original resolved QIDs. This seed uses those 47 manually-vetted original subjects only.

## Proposal-only extraction

| Stage | Count |
|---|---:|
| Raw relation candidates | 96 |
| Passed source gate | 96 |
| Passed v1 + v2 precision gates | 73 |
| Entities with >=1 accepted relation | 26 |
| Entities with >=2 predicate groups | 12 |

Predicate-group compositions: `{"developed_by+owned_by+used_for+uses+wikidata_p138_named_after+wikidata_p495_country_of_origin": 1, "developed_by+runs_on+used_for": 2, "developed_by+runs_on+used_for+uses": 1, "has_topic+published_by+wikidata_p407_language_of_work_or_name+wikidata_p495_country_of_origin": 1, "has_topic+wikidata_p495_country_of_origin": 1, "located_in+produces": 1, "located_in+product_of+runs_on+wikidata_p138_named_after+wikidata_p495_country_of_origin": 1, "part_of+wikidata_p527_has_part_s": 1, "used_for+uses": 1, "used_for+uses+wikidata_p461_opposite_of+wikidata_p527_has_part_s": 1, "wikidata_p282_writing_system+wikidata_p407_language_of_work_or_name": 1}`.

Properties without an existing schema mapping became a new predicate only when their Wikidata property occurred on at least three subjects. One-off and two-off properties remain quarantined with `wikidata_property_no_schema_match`; nothing in this report is promoted to serving memory.

## Lane comparison

arXiv: 0/331; Crossref: 46/100; OpenAlex: 2/6 paired; Wikidata: 12/47 manually-vetted original QIDs.

## Recommendation

Keep this output proposal-only. Review the accepted predicates and the explicit quarantine before any serving-memory promotion; promotion is a separate decision.
