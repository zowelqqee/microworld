# OpenAlex predicate-diversity reconnaissance

## Starting lane

The original OpenAlex quarantine contains seven unique relations: three
`supports`, two `enables`, one `uses`, and one `works_by`. They are abstract
sentence extractions, not durable API-level metadata. Several subjects are
explicitly discourse-like (`OpenMEE also`, `Textural evidence also`,
`Seminal`), so this seven-edge set is not itself a basis for promotion or a
multi-evidence cohort.

## What the existing snapshot retains

The stored normalized OpenAlex records retain only title, abstract text,
authors, DOI/source URL, publication date and topic bucket. They do not retain
the structured fields required for a graph-native diversity lane. Fresh reads
must therefore use the official OpenAlex Works API, not reconstruct facts from
the abstract fragments.

## API field check

The current official API documentation describes works, authors, institutions,
topics, publishers and their graph connections. Its LLM reference documents
`topics.id`, `authorships.institutions.id`, `is_oa`, and work singleton lookup.
The work-level API probe for DOI `10.1029/2012GC004370` returned:

- three named `topics` and a `primary_topic`;
- a non-empty `referenced_works` list of OpenAlex work IDs;
- author affiliations with the named institution Universitat Hamburg;
- `open_access` metadata.

Sources: <https://developers.openalex.org/> and
<https://developers.openalex.org/llms.txt>. The probe is reconnaissance only;
it created no relation artifacts.

## Diversity decision

Continue. The bounded extractor should create work-level `has_topic` and
`references_work` relations only when both endpoints are named by official API
metadata. This produces a `has_topic + references_work` predicate pair, which
is structurally distinct from Crossref's `created_by + published_by` pair:
topic classification and citation-graph edges are neither authorship nor
publisher metadata.

`authorships.institutions` and open-access status are deliberately not used in
the first pass: the former changes subject granularity (author -> institution)
and the latter is status-like rather than an explanatory graph relation. This
keeps the lane focused on a new, two-predicate work graph rather than volume.

