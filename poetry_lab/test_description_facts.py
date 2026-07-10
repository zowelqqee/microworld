"""Facts-layer checks: clause-local descriptive relations learned at ingest."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from poemcore.ingest import build_narrative_artifacts

# Repeated prepositional evidence ("на дверь", "в дом") is what admits a word
# as a noun head; the facts under test then require direct in-clause adjacency.
_CORPUS = """\
Он посмотрел на дверь и замолчал.
Она постучала в дверь.
Дверь в коридор скрипнула.
Он щурился на солнце.
Мы смотрели на солнце.
Солнце село за строенье.
Он вошёл в сад.
Она вернулась в сад.
За окнами стоял сад.
Она вернулась в город.
Он уехал за город в рощу.
Он поехал за город, в рощу.
"""


class DescriptionFactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.TemporaryDirectory()
        path = Path(cls._dir.name) / "mini.txt"
        path.write_text(_CORPUS, encoding="utf-8")
        cls.relations = build_narrative_artifacts(path)["description_relations"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def _rows(self, subject: str, kind: str) -> list[dict]:
        return [row for row in self.relations.get(subject, []) if row["kind"] == kind]

    def test_object_link_is_learned_from_in_clause_adjacency(self) -> None:
        links = self._rows("дверь", "object_link")
        self.assertIn(("в", "коридор"), [(row["predicate"], row["detail"]) for row in links])

    def test_event_place_keeps_an_agreeing_neuter_subject(self) -> None:
        events = self._rows("солнце", "event_place")
        self.assertIn(("село", "за строенье"), [(row["predicate"], row["detail"]) for row in events])

    def test_inverted_event_place_records_place_then_subject(self) -> None:
        events = self._rows("сад", "event_place_inverted")
        self.assertIn(("стоял", "за окнами"), [(row["predicate"], row["detail"]) for row in events])

    def test_an_oblique_subject_after_a_preposition_yields_no_link(self) -> None:
        # "за город в рощу" and "за город, в рощу" both mention the pair, but
        # "город" is an oblique run there, not a described subject.
        self.assertEqual(self._rows("город", "object_link"), [])


if __name__ == "__main__":
    unittest.main()
