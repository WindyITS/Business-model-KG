import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunker import read_and_chunk_file
from entity_resolver import resolve_entities
from evaluate_graph import _load_triples_from_json, evaluate
from llm_extractor import KnowledgeGraphExtraction, Triple


class PipelineComponentTests(unittest.TestCase):
    def test_entity_resolver_preserves_best_surface_form(self):
        extractions = [
            KnowledgeGraphExtraction(
                extraction_notes="ok",
                triples=[
                    Triple(
                        subject="OpenAI",
                        subject_type="Company",
                        relation="PARTNERS_WITH",
                        object="NASA",
                        object_type="Company",
                    ),
                    Triple(
                        subject="openai",
                        subject_type="Company",
                        relation="PARTNERS_WITH",
                        object="NASA",
                        object_type="Company",
                    ),
                ],
            )
        ]

        resolved = resolve_entities(extractions)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].subject, "OpenAI")

    def test_chunker_handles_local_files_without_tiktoken_network_access(self):
        chunks = read_and_chunk_file("data/microsoft_10k.txt")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_evaluator_accepts_resolved_triples_payload(self):
        payload = {
            "resolved_triples": [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Azure",
                    "object_type": "Offering",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "resolved.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            triples = _load_triples_from_json(str(path))

        self.assertEqual(len(triples), 1)

    def test_evaluate_scores_exact_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = {
                "triples": [
                    {
                        "subject": "Microsoft",
                        "subject_type": "Company",
                        "relation": "OFFERS",
                        "object": "Azure",
                        "object_type": "Offering",
                    }
                ]
            }
            gold_path = Path(tmp_dir) / "gold.json"
            pred_path = Path(tmp_dir) / "pred.json"
            gold_path.write_text(json.dumps(payload), encoding="utf-8")
            pred_path.write_text(json.dumps(payload), encoding="utf-8")

            gold = _load_triples_from_json(str(gold_path))
            predicted = _load_triples_from_json(str(pred_path))
            report = evaluate(predicted, gold)

        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
