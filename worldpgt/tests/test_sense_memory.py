from worldpgt.continuation.sense_memory import ExplicitSenseMemory


def test_builtin_senses_exist():
    memory = ExplicitSenseMemory()
    terms = memory.known_terms()
    for term in ["bank", "bat", "seal", "crane", "spring", "rock"]:
        assert term in terms

    bank_senses = {entry.sense_id for entry in memory.get_senses("bank")}
    assert bank_senses == {"financial_institution", "river_edge"}
    bat_senses = {entry.sense_id for entry in memory.get_senses("bat")}
    assert bat_senses == {"animal", "sports_equipment"}


def test_financial_bank_scores_higher_for_money_account_prompt():
    memory = ExplicitSenseMemory()
    scores = memory.score_senses("He put money into his account at the bank", "bank")
    assert scores["financial_institution"] > scores["river_edge"]
    assert scores["financial_institution"] == 2.0  # money + account


def test_river_bank_scores_higher_for_fisherman_river_prompt():
    memory = ExplicitSenseMemory()
    scores = memory.score_senses("The fisherman sat by the river near the bank", "bank")
    assert scores["river_edge"] > scores["financial_institution"]


def test_unknown_term_returns_empty_list():
    memory = ExplicitSenseMemory()
    assert memory.get_senses("table") == []
    assert memory.score_senses("a prompt about a table", "table") == {}


def test_no_cue_match_scores_zero():
    memory = ExplicitSenseMemory()
    scores = memory.score_senses("The man went to the bank to", "bank")
    assert scores["financial_institution"] == 0.0
    assert scores["river_edge"] == 0.0


def test_trust_scales_score():
    memory = ExplicitSenseMemory(include_builtin=False)
    memory.add_sense("port", "harbor", ["ship", "dock"], ["welcomed the ship"], trust=0.5)
    scores = memory.score_senses("the ship reached the dock near the port", "port")
    assert scores["harbor"] == 1.0  # 2 cues * 0.5 trust
