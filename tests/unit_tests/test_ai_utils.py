import os
import sys
import types
from unittest import mock

from validmind.ai import utils as ai_utils


def _reset_ai_utils_state():
    ai_utils.__dict__["__client"] = None
    ai_utils.__dict__["__model"] = None
    ai_utils.__dict__["__judge_llm"] = None
    ai_utils.__dict__["__judge_embeddings"] = None
    ai_utils.__dict__["__ack"] = None


def test_get_client_and_model_supports_gemini_env():
    _reset_ai_utils_state()

    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
        client, model = ai_utils.get_client_and_model()

    assert client is None
    assert model == ai_utils.GEMINI_MODEL


def test_get_judge_config_builds_gemini_models():
    _reset_ai_utils_state()

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGoogleGenerativeAIEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(
        ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings=FakeGoogleGenerativeAIEmbeddings,
    )

    with mock.patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-test-model"},
        clear=True,
    ), mock.patch.dict(sys.modules, {"langchain_google_genai": fake_module}):
        judge_llm, judge_embeddings = ai_utils.get_judge_config()

    assert isinstance(judge_llm, FakeChatGoogleGenerativeAI)
    assert judge_llm.kwargs == {
        "model": "gemini-test-model",
        "api_key": "test-key",
    }
    assert isinstance(judge_embeddings, FakeGoogleGenerativeAIEmbeddings)
    assert judge_embeddings.kwargs == {
        "model": ai_utils.GEMINI_EMBEDDINGS_MODEL,
        "google_api_key": "test-key",
    }


def test_is_configured_uses_resolved_judge_model():
    _reset_ai_utils_state()

    class FakeJudgeLLM:
        def invoke(self, messages):
            assert messages == [("user", "ping")]
            return types.SimpleNamespace(content="pong")

    with mock.patch.object(
        ai_utils,
        "get_judge_config",
        return_value=(FakeJudgeLLM(), None),
    ):
        assert ai_utils.is_configured() is True


def test_get_deepeval_model_supports_gemini_env():
    _reset_ai_utils_state()

    class FakeGeminiModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(GeminiModel=FakeGeminiModel)

    with mock.patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-test-model"},
        clear=True,
    ), mock.patch.dict(sys.modules, {"deepeval.models": fake_module}):
        model = ai_utils.get_deepeval_model()

    assert isinstance(model, FakeGeminiModel)
    assert model.kwargs == {
        "model": "gemini-test-model",
        "api_key": "test-key",
        "temperature": 0,
    }
