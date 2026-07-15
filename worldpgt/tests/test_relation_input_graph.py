from worldpgt.reasoning.relation_input_graph import default_relation_input_graph


def test_relation_input_graph_uses_denotation_edges_not_parser_templates():
    graph = default_relation_input_graph()
    assert graph.resolve("What does SpaceX make possible?", entity_spans=((10, 16),)) == "enables"
    assert graph.resolve("What capability does SpaceX provide?", entity_spans=((21, 27),)) == "enables"
    assert graph.resolve("What mechanism does SpaceX use?", entity_spans=((20, 26),)) == "works_by"
    assert graph.resolve("What is used by SpaceX?", entity_spans=((16, 22),)) == "uses"


def test_relation_input_graph_retains_all_denoted_edges_for_a_coordinated_question():
    graph = default_relation_input_graph()

    assert graph.resolve_all(
        "By whom was Adobe GoLive engineered, and for what application is Adobe GoLive employed?",
        entity_spans=((12, 24), (65, 77)),
    ) == ("developed_by", "used_for")
