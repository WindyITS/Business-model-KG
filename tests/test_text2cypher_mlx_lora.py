import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2cypher_mlx_lora import (
    build_mlx_lora_command,
    extract_json_dict,
    prepare_mlx_chat_dataset,
    score_prediction_payload,
)


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
    def test_prepare_mlx_chat_dataset_writes_train_and_test_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train_messages.jsonl"
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
                test_messages_path=test_path,
                output_root=output_root,
            )

            self.assertEqual(manifest["counts"]["train_rows"], 1)
            self.assertEqual(manifest["counts"]["test_rows"], 1)

            train_rows = [json.loads(line) for line in (output_root / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            test_rows = [json.loads(line) for line in (output_root / "test.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(train_rows[0]["id"], "train_row")
            self.assertEqual(test_rows[0]["id"], "test_row")
            self.assertEqual(train_rows[0]["messages"][-1]["role"], "assistant")
            self.assertEqual(test_rows[0]["metadata"]["example_id"], "test_row")

    def test_build_mlx_lora_command_uses_masked_chat_defaults(self):
        command = build_mlx_lora_command(
            model="google/gemma-4-E4B-it",
            data_dir=Path("/tmp/mlx_data"),
            adapter_path=Path("/tmp/adapters"),
            iters=123,
            batch_size=2,
            grad_accumulation_steps=8,
            num_layers=4,
            extra_args=("--seed", "7"),
        )

        self.assertEqual(command[:3], [sys.executable, "-m", "mlx_lm.lora"])
        self.assertIn("--train", command)
        self.assertIn("--mask-prompt", command)
        self.assertIn("--grad-checkpoint", command)
        self.assertIn("--iters", command)
        self.assertIn("123", command)
        self.assertEqual(command[-2:], ["--seed", "7"])

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


if __name__ == "__main__":
    unittest.main()
