# Schema expansion v1 — property audit

Дата аудита: 2026-07-18. Это только измерение: ни extraction, ни gate, ни serving/proposal overlay не изменялись.

## Scope and method

### Wikidata

Проверены **49 уже resolved QID**: 43 уникальных QID из `wikidata_density_recon/resolution_manifest.json` и 6 вручную approved QID из resolver-fix. Новые subjects не искались.

Для этих ровно 49 QID entity claims и labels повторно прочитаны через официальный Wikidata Action API. Из рассмотрения исключены structural/meta properties, применявшиеся в density recon (`P31`, `P279`, external IDs, URL/media и перечисленные там meta fields), а также все properties, уже представленные в текущей извлечённой/serving schema. Последняя группа включает, в частности, `P2283`, `P366`, `P306`, `P138`, `P17`, `P178`, `P495`, `P361`, `P282`, `P407`, `P131`, `P527`, `P921`, `P461`, `P400`, `P127`, `P1056`, `P176`, `P276`, `P123` и базовые density-recon mappings.

После этих исключений найдено **85** ещё не замапленных Wikidata property types. `entity-likelihood` ниже — доля observed values, которые являются `wikibase-item` (QID), а не time/string/free text. Приоритетный score = число subjects × эта доля; это намеренно предпочитает чистые entity-to-entity relations, которые могут увеличить connectivity.

### Crossref

Проверены артефакты для **46 promoted/unlocked canonical DOI** в `crossref_doi_seed_v1`. В них действительно сохранены:

- `frozen_entity_manifest.json`: DOI, title, уже извлечённые predicate groups;
- `proposal_relations.json`: только нормализованные `created_by` и `published_by` proposals;
- gate summary/accepted/rejected/quarantine и report.

Полных raw Crossref Works API JSON responses для этих DOI в workspace **не сохранено**. Поэтому поля исходного ответа (`subject`, `funder`, structured `license`, `container-title`, `volume`, `issue`, `page`, `reference-count`, `abstract`) невозможно честно посчитать как present/absent без новых API calls. По условию задачи новые Crossref calls не делались.

Это не означает, что этих полей в Crossref нет; это означает, что **нынешние retained artifacts их не позволяют аудитировать**. Единственное дополнительное структурированное поле, о котором можно говорить по текущим proposal artifacts, — DOI как identifier; оно не является содержательным graph predicate. `title`/evidence text — free text и не являются quick structured mapping.

## Wikidata: top-15 unmapped properties by subject frequency

| Property | Wikidata label | Subjects / 49 | Observed value form | Examples | Entity likelihood | Assessment |
|---|---|---:|---|---|---:|---|
| P571 | inception | 9 | time | 1981; 2024; 1987 | 0.00 | Clean scalar date, but not a connectivity relation. |
| P348 | software version identifier | 5 | string | `1.0`; `1.7.0`; `1.8.0` | 0.00 | Version metadata; not a graph-edge win. |
| P577 | publication date | 5 | time | 2021-03-19; 2024-11-20; 2015-03-12 | 0.00 | Clean scalar date, but not an entity relation. |
| P1433 | published in | 3 | QID | Nano Letters; *Computer*; *Philosophy of the Social Sciences* | 1.00 | Strong direct structured mapping candidate. |
| P1705 | native label | 3 | monolingual text | Bamm; Reml; Pomp | 0.00 | Name metadata, not a predicate edge. |
| P2093 | author name string | 3 | string | Srijit Goswami; Emre Mulazimoglu; Lieven M. K. Vandersypen | 0.00 | Free-name string; inferior to QID authorship. |
| P304 | page(s) | 3 | string | `2627-2632`; `12-16`; `501-524` | 0.00 | Bibliographic scalar, no entity endpoint. |
| P3878 | Soundex | 3 | string | B500; R540; P510 | 0.00 | Search/index metadata; exclude. |
| P3879 | Cologne phonetics | 3 | string | 16; 765; 161 | 0.00 | Search/index metadata; exclude. |
| P433 | issue | 3 | string | 4; 10; 4 | 0.00 | Bibliographic scalar, no entity endpoint. |
| P478 | volume | 3 | string | 15; 48; 37 | 0.00 | Bibliographic scalar, no entity endpoint. |
| P1072 | readable file format | 2 | QID | comma-separated values; tab-separated values; dBASE Table File Format Family | 1.00 | Clean QID edge; niche but direct. |
| P1073 | writable file format | 2 | QID | SPSS data file; SPSS output file; SPSS portable data format | 1.00 | Clean QID edge; niche but direct. |
| P1454 | legal form | 2 | QID | archives; spoločnosť s ručením obmedzeným | 1.00 | Structured but domain-specific/legal. |
| P1535 | used by | 2 | QID | traceroute; networking hardware | 1.00 | Clean inverse-style entity edge; semantic direction must be explicit. |

Other clean QID candidates at two subjects each: `P1542 has effect` (risk assessment; taxis; cell migration), `P1552 has characteristic` (instrumental album; evaluation strategy), `P2354 has list` (lists of large language models/open universities), `P277 programmed in` (Q2407; Q15777), and `P5805 OSI Model layer location` (network layer; application layer).

## Ranked candidate set

The first ranking uses score = frequency × clean-entity-likelihood. Frequency is small because this is a 49-QID cohort; equal-score rows are ordered by clarity and likely usefulness for compositional questions, not merely QID-ness.

| Rank | Candidate mapping | Source | Frequency | Clean entity likelihood | Score | Recommendation |
|---:|---|---|---:|---:|---:|---|
| 1 | `P1433 → published_in` | Wikidata | 3 | 1.00 | 3.0 | **Direct mapping.** Venue is a QID and values are clean named publications. |
| 2 | `P277 → programmed_in` | Wikidata | 2 | 1.00 | 2.0 | **Direct mapping.** QID programming-language endpoints; semantically crisp. |
| 3 | `P1535 → used_by` | Wikidata | 2 | 1.00 | 2.0 | **Direct mapping with explicit direction.** The subject is used by the object; do not silently invert it into existing `uses`. |
| 4 | `P1542 → has_effect` | Wikidata | 2 | 1.00 | 2.0 | **Direct mapping.** Structured QID targets; keep the broad causal wording visible in query templates. |
| 5 | `P1072 → readable_file_format` | Wikidata | 2 | 1.00 | 2.0 | **Direct mapping.** High precision, though narrow coverage/domain. |
| 6 | `P1073 → writable_file_format` | Wikidata | 2 | 1.00 | 2.0 | Direct mapping, same caveat as P1072. |
| 7 | `P5805 → osi_layer_location` | Wikidata | 2 | 1.00 | 2.0 | Direct mapping, but very domain-specific. |
| 8 | `P1454 → legal_form` | Wikidata | 2 | 1.00 | 2.0 | Technically direct; lower product/query priority due to legal-domain semantics. |
| 9 | `P1552 → has_characteristic` | Wikidata | 2 | 1.00 | 2.0 | Structured but semantically broad; validate question wording before adding. |
| 10 | `P2354 → has_list` | Wikidata | 2 | 1.00 | 2.0 | Structured but weakly useful for factual QA; low priority. |

### Quick wins versus non-wins

The top five (`published_in`, `programmed_in`, `used_by`, `has_effect`, `readable_file_format`) can be extracted using the same structured `property → predicate` approach as current Wikidata mappings. They need a reviewed field-to-predicate mapping and the existing gate discipline, but **no free-text parsing** and no new resolver work for their QID values.

`P571 inception`, `P577 publication date`, `P348 software version`, `P433 issue`, `P478 volume`, and `P304 page(s)` are clean scalar metadata, not free text, but do not create subject→entity graph connectivity. They are possible temporal/bibliographic schema additions, not answers to the current chain-density problem.

`P2093 author name string` and `P1705 native label` are strings/text rather than resolved entities; promoting them as relations would reintroduce the endpoint-quality problem. They are **not easier than the arXiv free-text-object problem**. Phonetic/index fields (`P3878`, `P3879`) should remain out of a semantic predicate schema.

## Answer to the practical question

There are real, previously unextracted structured Wikidata relations in the resolved cohort, but their observed coverage is modest: the best one appears on 3/49 subjects and the remaining clean candidates on 2/49. The best next low-engineering experiment is therefore a small, proposal-only mapping trial for `P1433`, `P277`, `P1535`, `P1542`, and `P1072`, measured for gate yield and newly connected entities. It is a genuine schema-expansion opportunity, not evidence that a large density jump is already available.

For Crossref, this audit cannot name a responsible new predicate from retained local raw data because the full DOI metadata payloads were not persisted. A later audit could re-fetch only the same 46 DOI records through the official API, but that is deliberately outside this task. It should not be inferred from this report that Crossref is structurally limited to authorship/publisher fields.
