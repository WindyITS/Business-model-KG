import unittest
from os import environ
from unittest.mock import patch

from runtime.model_provider import resolve_model_settings


class ModelProviderTests(unittest.TestCase):
    def test_local_provider_defaults_to_lm_studio_shape(self):
        settings = resolve_model_settings(provider="local")

        self.assertEqual(settings.provider, "local")
        self.assertEqual(settings.model, "local-model")
        self.assertEqual(settings.base_url, "http://localhost:1234/v1")
        self.assertEqual(settings.api_mode, "chat_completions")
        self.assertEqual(settings.api_key, "lm-studio")
        self.assertIsNone(settings.max_output_tokens)

    def test_opencode_go_normalizes_full_endpoint_url(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="kimi-k2.5",
            base_url="https://opencode.ai/zen/go/v1/chat/completions",
            api_key="secret",
        )

        self.assertEqual(settings.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(settings.api_mode, "chat_completions")
        self.assertEqual(settings.max_output_tokens, 20000)

    def test_opencode_go_normalizes_full_messages_endpoint_url(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="minimax-m2.7",
            base_url="https://opencode.ai/zen/go/v1/messages",
            api_key="secret",
        )

        self.assertEqual(settings.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(settings.api_mode, "messages")
        self.assertEqual(settings.max_output_tokens, 20000)

    def test_opencode_go_defaults_to_kimi(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            api_key="secret",
        )

        self.assertEqual(settings.model, "kimi-k2.5")
        self.assertEqual(settings.api_mode, "chat_completions")

    def test_opencode_go_reads_api_key_from_environment(self):
        with patch.dict(environ, {"OPENCODE_API_KEY": "env-secret"}, clear=True):
            settings = resolve_model_settings(
                provider="opencode-go",
                model="kimi-k2.5",
            )

        self.assertEqual(settings.api_key, "env-secret")
        self.assertEqual(settings.max_output_tokens, 20000)

    def test_opencode_go_honors_explicit_output_cap(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="kimi-k2.5",
            api_key="secret",
            max_output_tokens=1024,
        )

        self.assertEqual(settings.max_output_tokens, 1024)

    def test_opencode_go_accepts_mimo_as_explicit_override(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="mimo-v2-pro",
            api_key="secret",
        )

        self.assertEqual(settings.model, "mimo-v2-pro")
        self.assertEqual(settings.api_mode, "chat_completions")

    def test_opencode_go_accepts_human_friendly_model_aliases(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="MiniMax M2.7",
            api_key="secret",
        )

        self.assertEqual(settings.model, "minimax-m2.7")
        self.assertEqual(settings.api_mode, "messages")

    def test_opencode_go_accepts_prefixed_model_ids(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="opencode-go/kimi-k2.5",
            api_key="secret",
        )

        self.assertEqual(settings.model, "kimi-k2.5")
        self.assertEqual(settings.api_mode, "chat_completions")

    def test_opencode_go_rejects_unsupported_models(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_model_settings(
                provider="opencode-go",
                model="glm-5.1",
                api_key="secret",
            )

        self.assertIn("Unsupported opencode-go model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
