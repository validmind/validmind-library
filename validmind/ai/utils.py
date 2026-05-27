# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import importlib
import os

from openai import AzureOpenAI, OpenAI

from ..logging import get_logger
from ..utils import md_to_html

logger = get_logger(__name__)


__client = None
__model = None
__judge_llm = None
__judge_embeddings = None
OPENAI_MODEL = "gpt-4.1"
OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-small"
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_EMBEDDINGS_MODEL = "models/text-embedding-004"

# can be None, True or False (ternary to represent initial state, ack and failed ack)
__ack = None


class DescriptionFuture:
    """This will be immediately returned from generate_description so that
    the tests can continue to be run in parallel while the description is
    retrieved asynchronously.

    The value will be retrieved later and, if it is not ready yet, it should
    block until it is.
    """

    def __init__(self, future):
        self._future = future

    def get_description(self):
        if isinstance(self._future, tuple):
            description = self._future
        else:
            # This will block until the future is completed
            description = self._future.result()

        return md_to_html(description[0], mathml=True), description[1]


def _get_google_api_key():
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def _get_configured_provider():
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    if os.getenv("AZURE_OPENAI_KEY"):
        return "azure"

    return "gemini"


def get_client_and_model():
    """Get model and client to use for generating interpretations.

    On first call, it will look in the environment for the API key endpoint, model etc.
    and store them in a global variable to avoid loading them up again.
    """
    global __client, __model

    if __model is not None:
        return __client, __model

    provider = _get_configured_provider()

    if provider == "openai":
        __client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        __model = os.getenv("OPENAI_MODEL", OPENAI_MODEL)

        logger.debug(f"Using OpenAI {__model} for generating descriptions")

    elif provider == "azure":
        if "AZURE_OPENAI_ENDPOINT" not in os.environ:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT must be set to run LLM tests with Azure"
            )

        if "AZURE_OPENAI_MODEL" not in os.environ:
            raise ValueError(
                "AZURE_OPENAI_MODEL must be set to run LLM tests with Azure"
            )

        __client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2023-05-15"),
        )
        __model = os.getenv("AZURE_OPENAI_MODEL")

        logger.debug(f"Using Azure OpenAI {__model} for generating descriptions")

    else:
        __client = None
        __model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)

        logger.debug(f"Using Gemini {__model} for generating descriptions")

    return __client, __model


def _import_judge_dependencies():
    try:
        from langchain_core.embeddings import Embeddings
        from langchain_core.language_models.chat_models import BaseChatModel

        from validmind.models.function import FunctionModel
    except ImportError:
        raise ImportError("Please run `pip install validmind[llm]` to use LLM tests")

    return Embeddings, BaseChatModel, FunctionModel


def _unwrap_functional_judge_model(
    judge_model,
    expected_type,
    model_kind,
    FunctionModel,
):
    if not isinstance(judge_model, FunctionModel):
        return judge_model

    if isinstance(judge_model.model, expected_type):
        return judge_model.model

    raise ValueError(
        "The ValidMind Functional model provided does not have have a langchain "
        f"compatible {model_kind} model as a model attribute."
        "To use default ValidMind LLM, do not set judge_llm/judge_embedding parameter, "
        "ensure that you are connected to the ValidMind API and confirm ValidMind AI "
        "is enabled for your account."
    )


def _normalize_judge_overrides(
    judge_llm,
    judge_embeddings,
    Embeddings,
    BaseChatModel,
    FunctionModel,
):
    if judge_llm is None and judge_embeddings is None:
        return None

    judge_llm = _unwrap_functional_judge_model(
        judge_llm,
        BaseChatModel,
        "LLM",
        FunctionModel,
    )
    judge_embeddings = _unwrap_functional_judge_model(
        judge_embeddings,
        Embeddings,
        "embeddings",
        FunctionModel,
    )

    if (isinstance(judge_llm, BaseChatModel) or judge_llm is None) and (
        isinstance(judge_embeddings, Embeddings) or judge_embeddings is None
    ):
        return judge_llm, judge_embeddings

    raise ValueError(
        "Provided Judge LLM/Embeddings are not Langchain compatible. Ensure the judge "
        "LLM/embedding provided are an instance of Langchain BaseChatModel and "
        "LangchainEmbeddings.  To use default ValidMind LLM, do not set "
        "judge_llm/judge_embedding parameter, ensure that you are connected to the "
        "ValidMind API and confirm ValidMind AI is enabled for your account."
    )


def _build_gemini_judge_config(model):
    try:
        langchain_google_genai = importlib.import_module("langchain_google_genai")
    except ImportError:
        raise ImportError(
            "Please run `pip install validmind[llm]` to use Gemini LLM tests"
        )

    ChatGoogleGenerativeAI = getattr(langchain_google_genai, "ChatGoogleGenerativeAI")
    GoogleGenerativeAIEmbeddings = getattr(
        langchain_google_genai, "GoogleGenerativeAIEmbeddings"
    )
    google_api_key = _get_google_api_key()
    chat_kwargs = {"model": model}
    embeddings_kwargs = {
        "model": os.getenv("GEMINI_EMBEDDINGS_MODEL", GEMINI_EMBEDDINGS_MODEL),
    }

    if google_api_key:
        chat_kwargs["api_key"] = google_api_key
        embeddings_kwargs["google_api_key"] = google_api_key

    return (
        ChatGoogleGenerativeAI(**chat_kwargs),
        GoogleGenerativeAIEmbeddings(**embeddings_kwargs),
    )


def _build_openai_judge_config(client, model):
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        raise ImportError("Please run `pip install validmind[llm]` to use LLM tests")

    if client is not None and getattr(client, "base_url", None) is not None:
        os.environ["OPENAI_API_BASE"] = str(client.base_url)

    return (
        ChatOpenAI(api_key=client.api_key, model=model),
        OpenAIEmbeddings(api_key=client.api_key, model=OPENAI_EMBEDDINGS_MODEL),
    )


def _import_deepeval_base_llm():
    try:
        deepeval_base_model = importlib.import_module("deepeval.models.base_model")
    except ImportError:
        raise ImportError(
            "Please run `pip install validmind[llm]` to use Gemini DeepEval scorers"
        )

    return getattr(deepeval_base_model, "DeepEvalBaseLLM")


def _unwrap_deepeval_response(response):
    return getattr(response, "content", response)


def _build_gemini_deepeval_model(model):
    DeepEvalBaseLLM = _import_deepeval_base_llm()
    judge_llm, _ = _build_gemini_judge_config(model)

    class GeminiDeepEvalModel(DeepEvalBaseLLM):
        def __init__(self, chat_model, model_name):
            self._chat_model = chat_model
            self._model_name = model_name
            self.model = self.load_model()

        def load_model(self, *args, **kwargs):
            return self._chat_model

        def generate(self, prompt: str, schema=None):
            chat_model = self.load_model()
            if schema is not None and hasattr(chat_model, "with_structured_output"):
                response = chat_model.with_structured_output(schema).invoke(prompt)
            else:
                response = chat_model.invoke(prompt)

            return _unwrap_deepeval_response(response)

        async def a_generate(self, prompt: str, schema=None):
            chat_model = self.load_model()
            if schema is not None and hasattr(chat_model, "with_structured_output"):
                response = await chat_model.with_structured_output(schema).ainvoke(
                    prompt
                )
            else:
                response = await chat_model.ainvoke(prompt)

            return _unwrap_deepeval_response(response)

        def get_model_name(self, *args, **kwargs):
            return self._model_name

    return GeminiDeepEvalModel(judge_llm, model)


def get_judge_config(judge_llm=None, judge_embeddings=None):
    Embeddings, BaseChatModel, FunctionModel = _import_judge_dependencies()

    normalized_overrides = _normalize_judge_overrides(
        judge_llm,
        judge_embeddings,
        Embeddings,
        BaseChatModel,
        FunctionModel,
    )
    if normalized_overrides is not None:
        return normalized_overrides

    # grab default values if not passed at run time
    global __judge_llm, __judge_embeddings
    if __judge_llm and __judge_embeddings:
        return __judge_llm, __judge_embeddings

    provider = _get_configured_provider()
    client, model = get_client_and_model()

    if provider == "gemini":
        __judge_llm, __judge_embeddings = _build_gemini_judge_config(model)
    else:
        __judge_llm, __judge_embeddings = _build_openai_judge_config(client, model)

    return __judge_llm, __judge_embeddings


def get_deepeval_model():
    """Get the model object expected by DeepEval scorers.

    OpenAI/Azure scorers currently pass a model string. Gemini support requires a
    native DeepEval model object so the provider can be configured correctly.
    """

    provider = _get_configured_provider()
    _, model = get_client_and_model()

    if provider == "gemini":
        google_api_key = _get_google_api_key()
        if google_api_key is None:
            return _build_gemini_deepeval_model(model)

        try:
            deepeval_models = importlib.import_module("deepeval.models")
        except ImportError:
            raise ImportError(
                "Please run `pip install validmind[llm]` to use Gemini DeepEval scorers"
            )

        GeminiModel = getattr(deepeval_models, "GeminiModel")
        return GeminiModel(
            model=model,
            api_key=google_api_key,
            temperature=0,
        )

    return model


def set_judge_config(judge_llm, judge_embeddings):
    global __judge_llm, __judge_embeddings
    try:
        from langchain_core.embeddings import Embeddings
        from langchain_core.language_models.chat_models import BaseChatModel

        from validmind.models.function import FunctionModel
    except ImportError:
        raise ImportError("Please run `pip install validmind[llm]` to use LLM tests")
    if isinstance(judge_llm, BaseChatModel) and isinstance(
        judge_embeddings, Embeddings
    ):
        __judge_llm = judge_llm
        __judge_embeddings = judge_embeddings
        # Assuming 'your_object' is the object you want to check
    elif isinstance(judge_llm, FunctionModel) and isinstance(
        judge_embeddings, FunctionModel
    ):
        __judge_llm = judge_llm.model
        __judge_embeddings = judge_embeddings.model
    else:
        raise ValueError(
            "Provided Judge LLM/Embeddings are not Langchain compatible. Ensure the judge LLM/embedding provided are an instance of "
            "Langchain BaseChatModel and LangchainEmbeddings. To use default ValidMind LLM, do not set judge_llm/judge_embedding parameter, "
            "ensure that you are connected to the ValidMind API and confirm ValidMind AI is enabled for your account."
        )


def is_configured():
    global __ack

    if __ack:
        return True

    try:
        judge_llm, _ = get_judge_config()
        response = judge_llm.invoke(
            [("user", "ping")],
        )
        logger.debug(
            f"Received response from judge LLM: {getattr(response, 'content', response)}"
        )
        __ack = True
    except Exception as e:
        logger.debug(f"Failed to connect to judge LLM: {e}")
        __ack = False

    return __ack
