"""Offline test suite for groundrag.

Run:  python3 tests/test_rag.py
"""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

import rag  # noqa: E402

CORPUS = os.path.join(ROOT, "fixtures", "corpus")
EVAL = os.path.join(ROOT, "fixtures", "eval_set.json")


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = os.path.join(cls.tmp.name, "idx.db")
        cls.n = rag.ingest(CORPUS, cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- ingest -----------------------------------------------------

    def test_01_corpus_indexed(self):
        self.assertGreaterEqual(self.n, 3)  # 3 docs -> at least a chunk each

    # ---- retrieval ---------------------------------------------------

    def test_02_retrieve_finds_vacation_chunk(self):
        hits = rag.retrieve(self.db, "How many vacation days per month?")
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc, "handbook.md")
        self.assertIn("1.5 days per month", hits[0].text)

    def test_03_porter_stemming_matches_plural(self):
        # "fronds" in query; corpus has both frond/fronds
        hits = rag.retrieve(self.db, "yellow fronds meaning")
        self.assertTrue(any(h.doc == "plant_guide.md" for h in hits))

    # ---- abstain gate --------------------------------------------------

    def test_04_unanswerable_abstains(self):
        v = rag.ask(self.db, "recipe for banana bread")
        self.assertEqual(v.status, "abstain")
        self.assertIsNone(v.answer)

    def test_05_off_topic_abstains_even_with_some_overlap(self):
        # shares a stem with corpus ("company") but no real grounding
        v = rag.ask(self.db, "company revenue 2025 breakdown by region")
        self.assertEqual(v.status, "abstain")

    # ---- answer with citations -------------------------------------------

    def test_06_answerable_grounded_with_citation(self):
        v = rag.ask(self.db, "What are the support hours?")
        self.assertEqual(v.status, "grounded")
        self.assertTrue(any(c.startswith("product_notes.md#") for c in v.citations))
        self.assertIn("09:00 to 17:00", v.answer)

    def test_07_question_scaffolding_rewrite_recovers(self):
        # raw query padded with scaffolding; rewrite strips it and rehits
        v = rag.ask(self.db, "so tell me how often should the fern be repotted please")
        self.assertEqual(v.status, "grounded")
        self.assertIn("second spring", v.answer)

    # ---- eval harness integration -------------------------------------

    def test_08_eval_gates_pass_on_fixtures(self):
        import eval as ev
        result = ev.run_eval(self.db, verbose=False)
        self.assertEqual(len(result["hallucinations"]), 0)
        self.assertEqual(result["stats"]["answerable"][0],
                         result["stats"]["answerable"][1])
        self.assertEqual(result["stats"]["unanswerable"][0],
                         result["stats"]["unanswerable"][1])

    # ---- abstain log ---------------------------------------------------

    def test_09_abstain_log_written(self):
        log = os.path.join(self.tmp.name, "abstain.jsonl")
        v = rag.ask(self.db, "how tall is the Eiffel Tower")
        rag.log_abstain(log, v, "how tall is the Eiffel Tower")
        with open(log) as f:
            entry = json.loads(f.read().splitlines()[0])
        self.assertEqual(entry["query"], "how tall is the Eiffel Tower")
        self.assertIn("ts", entry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
