# Schema expansion v1 pipelines

Оба CLI создают только proposal artifacts. Они никогда не вызывают promotion
runner, не меняют accepted memory и не меняют serving overlay. После каждого
запуска `summary.json` имеет `review_state: ready_for_human_review`; решение о
promotion остаётся ручным.

## Wikidata

```bash
MICROWORLD_WIKI_USER_AGENT='your-contact' \
python3 -m worldpgt.knowledge_pump.wikidata_pipeline_v1 \
  --subjects-source main-dataset \
  --property-whitelist published_in,programmed_in,used_by,has_effect,readable_file_format \
  --output artifacts/schema_expansion_v1/wikidata_run_YYYYMMDD/ \
  --allow-network
```

`--subjects-source` принимает `main-dataset`, `unresolved-pool` или путь к JSON
list (строки либо objects с `subject`/`surface_subject`). `--property-whitelist`
принимает `default`, PIDs (`P1433,P277,...`) или указанные predicate names.
`--max-subjects N` задаёт строгую границу batch; `--delay-seconds` (default
0.15) ограничивает rate. Resolver использует existing exact + conservative
alias/P31 disambiguation и требует English Wikipedia anchor перед extraction.
`unresolved-pool` — полный текущий manifest без QID (не 30-item diagnostic
sample). Для overnight расширения используйте
`--property-whitelist first-round-plus-top5`: это bounded набор первого
content-bearing round плюс пять reviewed новых properties; P571 запрашивается
для auditable scalar/quarantine accounting, но не превращается в entity edge.

## Crossref

```bash
MICROWORLD_CROSSREF_USER_AGENT='your-contact' \
python3 -m worldpgt.knowledge_pump.crossref_pipeline_v1 \
  --dois artifacts/open_book_qa/crossref_doi_seed_v1/frozen_entity_manifest.json \
  --save-raw-responses \
  --output artifacts/schema_expansion_v1/crossref_run_YYYYMMDD/ \
  --allow-network
```

`--dois` — comma-separated DOI values либо JSON list/manifest. Каждый DOI
запрашивается ровно один раз через официальный Crossref Works API; нет search
expansion. С `--save-raw-responses` полный API envelope записывается в
`raw_responses/`, так что будущий field audit может читать исходные поля.
`--delay-seconds` (default 0.5) ограничивает rate.

## Outputs and safe re-runs

Каждый output directory получает `raw_candidates.json`, `proposal_overlay.json`,
`rejected.json`, `quarantine.json`, и `summary.json`; Wikidata дополнительно
пишет `resolution_manifest.json`. Перед gate обе команды сравнивают edge keys с
текущим composed serving graph и удаляют совпадения. Поэтому summary содержит
`already_promoted_overlap_filtered` и всегда имеет
`proposal_relation_overlap_with_serving: 0`.

Для offline проверки wiring используйте `--dry-run` (можно вместе с маленьким
input): он не делает network calls, но создаёт полный пустой proposal package
и summary. Для реального retrieval дополнительно обязателен `--allow-network`
и соответствующий User-Agent environment variable.
