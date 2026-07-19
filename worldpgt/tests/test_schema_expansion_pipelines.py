from __future__ import annotations

import json

from worldpgt.knowledge_pump.crossref_pipeline_v1 import load_dois, run_pipeline as run_crossref
from worldpgt.knowledge_pump.wikidata_pipeline_v1 import run_pipeline as run_wikidata


def _claim(qid: str) -> dict:
    return {"mainsnak": {"datavalue": {"value": {"id": qid}}}}


class _WikidataClient:
    calls = 0

    def search(self, _label: str) -> list[dict]:
        self.calls += 1
        return [{"id": "Q1", "label": "Alpha", "display": {"label": {"language": "en"}}}]

    def entities(self, qids: list[str], *, properties: str) -> dict[str, dict]:
        self.calls += 1
        if properties == "labels":
            return {"Q2": {"labels": {"en": {"value": "Journal"}}}}
        return {"Q1": {"labels": {"en": {"value": "Alpha"}}, "sitelinks": {"enwiki": {"title": "Alpha"}}, "claims": {"P1433": [_claim("Q2")]}}}


class _CrossrefClient:
    calls = 0

    def work(self, doi: str) -> dict:
        self.calls += 1
        return {"status": "ok", "message": {"DOI": doi, "title": ["A Work"], "publisher": "Press", "author": [{"given": "Ada", "family": "Lovelace"}]}}


def test_wikidata_pipeline_resolves_extracts_gates_and_filters_serving_overlap():
    result = run_wikidata(
        [{"subject": "Alpha", "surface_subject": "Alpha"}], property_ids=("P1433",),
        client=_WikidataClient(), serving_rows=[],
    )
    assert result["overlap"] == 0
    assert result["gate"]["passed_precision_gate"] == 1
    assert result["gate"]["accepted_proposal_overlay"][0]["predicate"] == "published_in"
    repeat = run_wikidata(
        [{"subject": "Alpha", "surface_subject": "Alpha"}], property_ids=("P1433",),
        client=_WikidataClient(), serving_rows=result["candidates"],
    )
    assert repeat["overlap"] == 1
    assert repeat["gate"]["passed_precision_gate"] == 0


def test_crossref_pipeline_caches_full_response_and_filters_serving_overlap(tmp_path):
    cache = tmp_path / "raw"
    result = run_crossref(["10.1000/example"], client=_CrossrefClient(), save_raw_dir=cache, serving_rows=[])
    assert result["gate"]["passed_precision_gate"] == 2
    cached = json.loads(next(cache.glob("*.json")).read_text())
    assert cached["status"] == "ok"
    repeat = run_crossref(["10.1000/example"], client=_CrossrefClient(), save_raw_dir=None, serving_rows=result["candidates"])
    assert repeat["overlap"] == 2
    assert repeat["gate"]["passed_precision_gate"] == 0


def test_crossref_doi_loader_accepts_existing_manifest(tmp_path):
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps([{"canonical_doi": "10.1/A"}, "10.1/b", {"other": "ignored"}]))
    assert load_dois(str(source)) == ["10.1/a", "10.1/b"]
