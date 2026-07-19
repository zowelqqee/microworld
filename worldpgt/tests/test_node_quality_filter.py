from pathlib import Path

from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.node_quality_filter import assess_node_quality


def _index(tmp_path: Path) -> EntitySurfaceIndex:
    absent = tmp_path / "absent.json"
    return EntitySurfaceIndex(absent, absent, absent)


def test_blocks_authorial_subject(tmp_path: Path) -> None:
    result = assess_node_quality("this paper", "NB2Slides", "this paper presents NB2Slides.", _index(tmp_path))
    assert not result.accepted
    assert "authorial_self_reference" in result.reasons


def test_blocks_generic_event_object(tmp_path: Path) -> None:
    result = assess_node_quality("NB2Slides", "identifying vulnerabilities", "NB2Slides helps identifying vulnerabilities.", _index(tmp_path))
    assert not result.accepted
    assert "event_like_node" in result.reasons


def test_flags_colon_list_inference(tmp_path: Path) -> None:
    sentence = "Architecture: Quantum Neural Networks for sensor fusion, Nav-Q for navigation."
    result = assess_node_quality("Nav-Q", "navigation", sentence, _index(tmp_path))
    assert not result.accepted
    assert "list_derived_context" in result.reasons


def test_accepts_two_resolved_named_nodes(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.json"
    overlay.write_text(
        '[{"overlay_type":"overlay_entity","label":"SciServer"},'
        '{"overlay_type":"overlay_entity","label":"SkyServer"}]',
        encoding="utf-8",
    )
    index = EntitySurfaceIndex(overlay, tmp_path / "absent.json", tmp_path / "also_absent.json")
    result = assess_node_quality("SciServer", "SkyServer", "SciServer builds upon SkyServer.", index)
    assert result.accepted
    assert result.reasons == ()
