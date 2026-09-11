from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from validmind.tests.model_validation.ragas.utils import _ragas_is_finished_parser


def test_ragas_is_finished_parser_accepts_gemini_stop_reason():
    response = SimpleNamespace(
        flatten=lambda: [
            SimpleNamespace(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(
                                content="done",
                                response_metadata={"finish_reason": "STOP"},
                            )
                        )
                    ]
                ]
            )
        ]
    )

    assert _ragas_is_finished_parser(response) is True


def test_ragas_is_finished_parser_accepts_max_tokens_reason():
    response = SimpleNamespace(
        flatten=lambda: [
            SimpleNamespace(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(
                                content="partial",
                                response_metadata={"finish_reason": "MAX_TOKENS"},
                            )
                        )
                    ]
                ]
            )
        ]
    )

    assert _ragas_is_finished_parser(response) is True
