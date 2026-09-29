"""Shared fixtures and fake SDK doubles for the offline test suite.

No test in this project ever performs a real LLM API call: the OpenAI SDK is
replaced by injected fakes and the provider by the deterministic mock client.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.llm_client import MockLLMClient
from src.processor import InquiryProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "sample_inquiries.csv"


def make_response(payload, finish_reason: str = "stop", refusal: str | None = None, usage=None):
    """Build a minimal stand-in for an OpenAI chat-completion response."""
    content = payload if isinstance(payload, str) else json.dumps(payload)
    message = SimpleNamespace(content=content, refusal=refusal)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=usage
    )
    return response


class FakeCompletions:
    """Records create() calls and replays scripted behaviors in order."""

    def __init__(self, behaviors: list):
        self._behaviors = list(behaviors)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._behaviors:
            raise AssertionError("unexpected extra SDK call")
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        if isinstance(behavior, (dict, str)):
            return make_response(behavior)
        return behavior


class FakeOpenAIChat:
    """Stand-in for openai.OpenAI exposing only what OpenAIClient touches."""

    def __init__(self, behaviors: list):
        self.chat = SimpleNamespace(completions=FakeCompletions(behaviors))


def make_rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after else None
    response = httpx.Response(429, headers=headers, request=request)
    return openai.RateLimitError("rate limit exceeded", response=response, body=None)


def make_auth_error() -> openai.AuthenticationError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return openai.AuthenticationError("invalid api key", response=response, body=None)


def make_connection_error() -> openai.APIConnectionError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.APIConnectionError(request=request)


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def processor(mock_client: MockLLMClient) -> InquiryProcessor:
    return InquiryProcessor(mock_client)


@pytest.fixture
def sample_inquiries():
    from src.csv_loader import load_inquiries

    return load_inquiries(SAMPLE_CSV)


# -- upgraded-stack fixtures (service / storage / API) ---------------------------


@pytest.fixture
def store(tmp_path):
    from src.storage import InquiryStore

    inquiry_store = InquiryStore(tmp_path / "test.db")
    inquiry_store.initialize()
    return inquiry_store


@pytest.fixture
def service(store):
    from src.service import InquiryService

    return InquiryService(
        llm_client=MockLLMClient(), store=store, provider="mock", model="mock-1"
    )


@pytest.fixture
def api_client(service):
    from fastapi.testclient import TestClient

    from src.api import create_app

    return TestClient(create_app(service=service))
