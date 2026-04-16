import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from text2cypher.mlx import (
    build_mlx_lora_command,
    extract_json_dict,
    load_mlx_model_and_tokenizer,
    normalize_cypher_semantic,
    prepare_mlx_chat_dataset,
    score_prediction_payload,
)
from text2cypher.cli.prepare_text2cypher_mlx_dataset import main as prepare_cli_main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _messages_row(*, example_id: str, split: str, prompt: str, completion: str) -> dict:
    return {
        "sft_example_id": example_id,
        "training_example_ids": [example_id],
        "split": split,
        "prompt": prompt,
        "completion": completion,
        "messages": [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "metadata": {
            "example_id": example_id,
            "intent_id": example_id,
            "family_id": "QF00",
            "answerable": True,
            "refusal_reason": None,
            "result_shape": [{"column": "company", "type": "string"}],
        },
    }


class Text2CypherMlxLoraTests(unittest.TestCase):
    def test_prepare_mlx_chat_dataset_writes_train_valid_and_test_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train_messages.jsonl"
            valid_path = tmp_root / "valid_messages.jsonl"
            test_path = tmp_root / "test_messages.jsonl"
            output_root = tmp_root / "mlx_data"

            _write_jsonl(
                train_path,
                [
                    _messages_row(
                        example_id="train_row",
                        split="train",
                        prompt="Which company serves developers?",
                        completion='{"answerable":true,"cypher":"MATCH ...","params":{"customer_type":"developers"}}',
                    )
                ],
            )
            _write_jsonl(
                valid_path,
                [
                    _messages_row(
                        example_id="valid_row",
                        split="valid",
                        prompt="Which company partners with Acme?",
                        completion='{"answerable":true,"cypher":"MATCH ...","params":{"company":"Acme"}}',
                    )
                ],
            )
            _write_jsonl(
                test_path,
                [
                    _messages_row(
                        example_id="test_row",
                        split="heldout_test",
                        prompt="Which companies operate in APAC?",
                        completion='{"answerable":true,"cypher":"MATCH ...","params":{"place":"APAC"}}',
                    )
                ],
            )

            manifest = prepare_mlx_chat_dataset(
                train_messages_path=train_path,
                valid_messages_path=valid_path,
                test_messages_path=test_path,
                output_root=output_root,
            )

            self.assertEqual(manifest["counts"]["train_rows"], 1)
            self.assertEqual(manifest["counts"]["valid_rows"], 1)
            self.assertEqual(manifest["counts"]["test_rows"], 1)

            train_rows = [json.loads(line) for line in (output_root / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            valid_rows = [json.loads(line) for line in (output_root / "valid.jsonl").read_text(encoding="utf-8").splitlines()]
            test_rows = [json.loads(line) for line in (output_root / "test.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(train_rows[0]["id"], "train_row")
            self.assertEqual(valid_rows[0]["id"], "valid_row")
            self.assertEqual(test_rows[0]["id"], "test_row")
            self.assertEqual(train_rows[0]["messages"][-1]["role"], "assistant")
            self.assertEqual(valid_rows[0]["split"], "valid")
            self.assertEqual(test_rows[0]["metadata"]["example_id"], "test_row")

    def test_build_mlx_lora_command_uses_masked_chat_defaults(self):
        command = build_mlx_lora_command(
            model="Qwen/Qwen3-4B",
            data_dir=Path("/tmp/mlx_data"),
            adapter_path=Path("/tmp/adapters"),
            iters=123,
            batch_size=2,
            grad_accumulation_steps=8,
            num_layers=4,
            extra_args=("--seed", "7"),
        )

        self.assertEqual(command[:4], [sys.executable, "-m", "mlx_lm", "lora"])
        self.assertIn("--train", command)
        self.assertIn("--mask-prompt", command)
        self.assertIn("--grad-checkpoint", command)
        self.assertIn("--iters", command)
        self.assertIn("123", command)
        self.assertEqual(command[-2:], ["--seed", "7"])

    def test_prepare_cli_passes_valid_messages_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train_messages.jsonl"
            valid_path = tmp_root / "valid_messages.jsonl"
            test_path = tmp_root / "test_messages.jsonl"
            output_root = tmp_root / "mlx_data"

            _write_jsonl(
                train_path,
                [_messages_row(example_id="train_row", split="train", prompt="train", completion='{"answerable":false,"reason":"x"}')],
            )
            _write_jsonl(
                valid_path,
                [_messages_row(example_id="valid_row", split="valid", prompt="valid", completion='{"answerable":false,"reason":"x"}')],
            )
            _write_jsonl(
                test_path,
                [_messages_row(example_id="test_row", split="heldout_test", prompt="test", completion='{"answerable":false,"reason":"x"}')],
            )

            with mock.patch("sys.stdout.write"):
                exit_code = prepare_cli_main(
                    [
                        "--train-messages-path",
                        str(train_path),
                        "--valid-messages-path",
                        str(valid_path),
                        "--test-messages-path",
                        str(test_path),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "valid.jsonl").exists())
            manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["train_rows"], 1)
            self.assertEqual(manifest["counts"]["valid_rows"], 1)
            self.assertEqual(manifest["counts"]["test_rows"], 1)

    def test_extract_json_dict_recovers_final_json_object_from_reasoning_text(self):
        text = "<think>Need to reason a bit.</think>\n{\"answerable\":false,\"reason\":\"not_in_graph\"}"
        parsed, extracted = extract_json_dict(text)
        self.assertEqual(parsed, {"answerable": False, "reason": "not_in_graph"})
        self.assertEqual(extracted, '{"answerable":false,"reason":"not_in_graph"}')

    def test_score_prediction_payload_normalizes_cypher_whitespace_and_param_order(self):
        gold_completion = (
            '{"answerable":true,"cypher":"MATCH (c:Company {name: $company}) RETURN c.name AS company",'
            '"params":{"company":"Acme"}}'
        )
        prediction = (
            '{"params":{"company":"Acme"},'
            '"cypher":"MATCH   (c:Company {name: $company})   RETURN c.name AS company",'
            '"answerable":true}'
        )

        metrics = score_prediction_payload(prediction, gold_completion=gold_completion)
        self.assertTrue(metrics["valid_json"])
        self.assertTrue(metrics["answerable_match"])
        self.assertTrue(metrics["params_exact_match"])
        self.assertFalse(metrics["cypher_exact_match"])
        self.assertTrue(metrics["cypher_normalized_match"])
        self.assertTrue(metrics["structured_match"])

    def test_load_mlx_model_and_tokenizer_supports_base_model_without_adapter(self):
        fake_load = mock.Mock(return_value=("model", "tokenizer"))
        fake_generate = object()
        fake_module = types.SimpleNamespace(load=fake_load, generate=fake_generate)

        with mock.patch.dict(sys.modules, {"mlx_lm": fake_module}):
            model, tokenizer, generate_fn = load_mlx_model_and_tokenizer(
                model_path="Qwen/Qwen3-4B",
                adapter_path=None,
            )

        fake_load.assert_called_once_with("Qwen/Qwen3-4B")
        self.assertEqual(model, "model")
        self.assertEqual(tokenizer, "tokenizer")
        self.assertIs(generate_fn, fake_generate)

    def test_normalize_cypher_semantic_allows_equivalent_qf19_direct_membership(self):
        gold = (
            "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->"
            "(ct:CustomerType {name: $customer_type}) "
            "MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) "
            "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
            "WHERE s.company_name = c.name AND o.company_name = c.name "
            "RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment"
        )
        predicted = (
            "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: c.name})-[:SERVES]->"
            "(:CustomerType {name: $customer_type}) "
            "MATCH (s)-[:SELLS_THROUGH]->(:Channel {name: $channel}) "
            "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
            "WHERE s.company_name = c.name AND o.company_name = c.name "
            "RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment"
        )

        self.assertEqual(normalize_cypher_semantic(predicted), normalize_cypher_semantic(gold))

    def test_score_prediction_payload_reports_semantic_match_for_equivalent_qf19(self):
        gold_completion = (
            '{"answerable":true,"cypher":"MATCH (c:Company)-[:HAS_SEGMENT]->'
            '(s:BusinessSegment)-[:SERVES]->(ct:CustomerType {name: $customer_type}) '
            'MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) '
            'MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) '
            'WHERE s.company_name = c.name AND o.company_name = c.name '
            'RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment",'
            '"params":{"channel":"direct sales","customer_type":"IT professionals","offering":"Dovetail Core"}}'
        )
        prediction = (
            '{"answerable":true,"cypher":"MATCH (c:Company)-[:HAS_SEGMENT]->'
            '(s:BusinessSegment {company_name: c.name})-[:SERVES]->(:CustomerType {name: $customer_type}) '
            'MATCH (s)-[:SELLS_THROUGH]->(:Channel {name: $channel}) '
            'MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) '
            'WHERE s.company_name = c.name AND o.company_name = c.name '
            'RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment",'
            '"params":{"customer_type":"IT professionals","channel":"direct sales","offering":"Dovetail Core"}}'
        )

        metrics = score_prediction_payload(prediction, gold_completion=gold_completion)
        self.assertFalse(metrics["structured_match"])
        self.assertFalse(metrics["cypher_normalized_match"])
        self.assertTrue(metrics["cypher_semantic_match"])
        self.assertTrue(metrics["structured_semantic_match"])

    def test_score_prediction_payload_keeps_descendant_qf19_mismatch_as_non_semantic(self):
        gold_completion = (
            '{"answerable":true,"cypher":"MATCH (c:Company)-[:HAS_SEGMENT]->'
            '(s:BusinessSegment)-[:SERVES]->(ct:CustomerType {name: $customer_type}) '
            'MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) '
            'MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) '
            'WHERE s.company_name = c.name AND o.company_name = c.name '
            'RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment",'
            '"params":{"channel":"direct sales","customer_type":"IT professionals","offering":"Dovetail Core"}}'
        )
        prediction = (
            '{"answerable":true,"cypher":"MATCH (c:Company)-[:HAS_SEGMENT]->'
            '(s:BusinessSegment {company_name: c.name})-[:SERVES]->(:CustomerType {name: $customer_type}) '
            'MATCH (s)-[:SELLS_THROUGH]->(:Channel {name: $channel}) '
            'MATCH (s)-[:OFFERS]->(root:Offering {company_name: c.name}) '
            'MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: c.name, name: $offering}) '
            'RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment",'
            '"params":{"customer_type":"IT professionals","channel":"direct sales","offering":"Dovetail Core"}}'
        )

        metrics = score_prediction_payload(prediction, gold_completion=gold_completion)
        self.assertFalse(metrics["cypher_semantic_match"])
        self.assertFalse(metrics["structured_semantic_match"])

    def test_normalize_cypher_semantic_allows_company_alias_variation_in_qf19(self):
        gold = (
            "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->"
            "(ct:CustomerType {name: $customer_type}) "
            "MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) "
            "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
            "WHERE s.company_name = c.name AND o.company_name = c.name "
            "RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment"
        )
        predicted = (
            "MATCH (company:Company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: company.name})-[:SERVES]->"
            "(:CustomerType {name: $customer_type}) "
            "MATCH (s)-[:SELLS_THROUGH]->(:Channel {name: $channel}) "
            "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
            "WHERE s.company_name = company.name AND o.company_name = company.name "
            "RETURN DISTINCT company.name AS company, s.name AS segment ORDER BY company, segment"
        )

        self.assertEqual(normalize_cypher_semantic(predicted), normalize_cypher_semantic(gold))

    def test_score_prediction_payload_reports_semantic_match_for_company_alias_variation(self):
        gold_completion = (
            '{"answerable":true,"cypher":"MATCH (c:Company)-[:HAS_SEGMENT]->'
            '(s:BusinessSegment)-[:SERVES]->(ct:CustomerType {name: $customer_type}) '
            'MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) '
            'MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) '
            'WHERE s.company_name = c.name AND o.company_name = c.name '
            'RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment",'
            '"params":{"channel":"direct sales","customer_type":"IT professionals","offering":"Dovetail Core"}}'
        )
        prediction = (
            '{"answerable":true,"cypher":"MATCH (company:Company)-[:HAS_SEGMENT]->'
            '(s:BusinessSegment {company_name: company.name})-[:SERVES]->(:CustomerType {name: $customer_type}) '
            'MATCH (s)-[:SELLS_THROUGH]->(:Channel {name: $channel}) '
            'MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) '
            'WHERE s.company_name = company.name AND o.company_name = company.name '
            'RETURN DISTINCT company.name AS company, s.name AS segment ORDER BY company, segment",'
            '"params":{"customer_type":"IT professionals","channel":"direct sales","offering":"Dovetail Core"}}'
        )

        metrics = score_prediction_payload(prediction, gold_completion=gold_completion)
        self.assertTrue(metrics["cypher_semantic_match"])
        self.assertTrue(metrics["structured_semantic_match"])


if __name__ == "__main__":
    unittest.main()
