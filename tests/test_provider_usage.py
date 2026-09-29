"""Provider selection and usage-metadata tests (all offline)."""

from types import SimpleNamespace

import pytest

from src.config import ConfigError, load_config
from src.llm_client import MockLLMClient, OpenAIClient, build_llm_client
from src.models import TokenUsage
from tests.conftest import FakeOpenAIChat, make_rate_limit_error, make_response


class TestProviderSelection:
    def test_build_llm_client_mock(self):
        config = load_config(provider="mock")
        assert isinstance(build_llm_client(config), MockLLMClient)

    def test_build_llm_client_openai_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = load_config(provider="openai")
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            build_llm_client(config)

    def test_build_llm_client_openai_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = load_config(provider="openai", model="gpt-4o-mini")
        assert isinstance(build_llm_client(config), OpenAIClient)

    def test_unknown_provider_rejected_by_config(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "claude")  # not configured yet
        with pytest.raises(ConfigError, match="unknown provider"):
            load_config()

    def test_llm_provider_env_var_selects_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        assert load_config().provider == "mock"

    def test_explicit_argument_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        assert load_config(provider="mock").provider == "mock"


class TestAnalyzeDetailedMetadata:
    def test_openai_client_reports_attempts_latency_usage(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        fake = FakeOpenAIChat(
            [make_rate_limit_error(), make_response(
                {"summary": "s", "category": "Sales", "priority": "Low"}, usage=usage
            )]
        )
        client = OpenAIClient(
            api_key="k", model="gpt-4o-mini", sdk_client=fake,
            retry_initial_delay=0.5, sleep=lambda _s: None,
        )
        result = client.analyze_detailed("Alice", "hi")
        assert result.payload["category"] == "Sales"
        assert result.attempts == 2
        assert result.latency_ms >= 0
        assert result.usage == TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)

    def test_openai_client_without_usage_reports_none(self):
        fake = FakeOpenAIChat([make_response({"summary": "s", "category": "Sales", "priority": "Low"})])
        client = OpenAIClient(api_key="k", model="m", sdk_client=fake, sleep=lambda _s: None)
        result = client.analyze_detailed("Alice", "hi")
        assert result.usage is None
        assert result.attempts == 1

    def test_retry_after_header_overrides_backoff_delay(self):
        fake = FakeOpenAIChat(
            [make_rate_limit_error(retry_after="5"), make_response(
                {"summary": "s", "category": "Sales", "priority": "Low"}
            )]
        )
        sleeps: list[float] = []
        client = OpenAIClient(
            api_key="k", model="m", sdk_client=fake,
            retry_initial_delay=1.0, retry_max_delay=2.0, sleep=sleeps.append,
        )
        client.analyze("Alice", "hi")
        assert sleeps == [5.0]  # provider hint (5s) beats computed delay (1s)

    def test_non_numeric_retry_after_is_ignored(self):
        fake = FakeOpenAIChat(
            [make_rate_limit_error(retry_after="Wed, 21 Oct 2026 07:28:00 GMT"),
             make_response({"summary": "s", "category": "Sales", "priority": "Low"})]
        )
        sleeps: list[float] = []
        client = OpenAIClient(
            api_key="k", model="m", sdk_client=fake,
            retry_initial_delay=1.5, retry_max_delay=10.0, sleep=sleeps.append,
        )
        client.analyze("Alice", "hi")
        assert sleeps == [1.5]

    def test_mock_client_reports_attempts_and_null_usage(self):
        result = MockLLMClient().analyze_detailed("Alice", "hi")
        assert result.attempts == 1
        assert result.usage is None
        assert result.payload["category"] == "General Question"
