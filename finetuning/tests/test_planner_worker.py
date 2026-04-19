import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.planner_worker import PlannerGenerator


class PlannerWorkerTests(unittest.TestCase):
    def test_generate_uses_sampler_api_and_optional_adapter(self):
        load = Mock(return_value=("model", types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "PROMPT")))
        generate = Mock(return_value='{"answerable":true}')
        make_sampler = Mock(return_value="SAMPLER")

        fake_generate_module = types.SimpleNamespace(generate=generate)
        fake_sample_utils_module = types.SimpleNamespace(make_sampler=make_sampler)
        fake_utils_module = types.SimpleNamespace(load=load)

        original_modules = {}
        for name, module in {
            "mlx_lm.generate": fake_generate_module,
            "mlx_lm.sample_utils": fake_sample_utils_module,
            "mlx_lm.utils": fake_utils_module,
        }.items():
            original_modules[name] = sys.modules.get(name)
            sys.modules[name] = module

        try:
            generator = PlannerGenerator(
                model_path="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                adapter_path=None,
            )
            result = generator.generate("Which companies partner with Dell?", max_tokens=256)
        finally:
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(result, '{"answerable":true}')
        make_sampler.assert_called_once_with(temp=0.0)
        load.assert_called_once_with(
            "mlx-community/Qwen3-4B-Instruct-2507-4bit",
            tokenizer_config={"trust_remote_code": True},
        )
        kwargs = generate.call_args.kwargs
        self.assertEqual(kwargs["sampler"], "SAMPLER")
        self.assertEqual(kwargs["max_tokens"], 256)
        self.assertNotIn("temp", kwargs)


if __name__ == "__main__":
    unittest.main()
