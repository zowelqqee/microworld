from __future__ import annotations

import unittest

from poemcore.transitions import qa_depth_one


class QADepthOneAdapterTest(unittest.TestCase):
    def test_production_rule_output_enters_shared_hypothesis_lifecycle(self) -> None:
        fixture = [
            {"overlay_type": "overlay_relation", "subject": "A", "predicate": "owned_by", "object": "B"},
            {"overlay_type": "overlay_relation", "subject": "B", "predicate": "owned_by", "object": "C"},
        ]
        evaluations = qa_depth_one(fixture)

        self.assertEqual(len(evaluations), 1)
        self.assertTrue(evaluations[0].transition.accepted)
        self.assertEqual(evaluations[0].hypothesis.delta.assertions[0].triple, ("A", "owned_by", "C"))
        self.assertEqual(evaluations[0].hypothesis.proof_chain, ("ownership_transitivity_v1",))


if __name__ == "__main__":
    unittest.main()
