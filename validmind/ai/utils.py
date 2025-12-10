# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os

from openai import AzureOpenAI, OpenAI

from ..logging import get_logger
from ..utils import md_to_html

logger = get_logger(__name__)


__client = None
__model = None
__judge_llm = None
__judge_embeddings = None
EMBEDDINGS_MODEL = "text-embedding-3-small"

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


def get_client_and_model():
    """Get model and client to use for generating interpretations.

    On first call, it will look in the environment for the API key endpoint, model etc.
    and store them in a global variable to avoid loading them up again.
    """
    global __client, __model

    if __client and __model:
        return __client, __model

    if "OPENAI_API_KEY" in os.environ:
        __client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        __model = os.getenv("VM_OPENAI_MODEL", "gpt-4o")

        logger.debug(f"Using OpenAI {__model} for generating descriptions")

    elif "AZURE_OPENAI_KEY" in os.environ:
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
        raise ValueError(
            "OPENAI_API_KEY, AZURE_OPENAI_KEY must be setup to use LLM features"
        )

    return __client, __model


def get_judge_config(judge_llm=None, judge_embeddings=None):
    try:
        from langchain_core.embeddings import Embeddings
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        from validmind.models.function import FunctionModel
    except ImportError:
        raise ImportError("Please run `pip install validmind[llm]` to use LLM tests")

    if judge_llm is not None or judge_embeddings is not None:
        if isinstance(judge_llm, FunctionModel) and judge_llm is not None:
            if isinstance(judge_llm.model, BaseChatModel):
                judge_llm = judge_llm.model
            else:
                raise ValueError(
                    "The ValidMind Functional model provided does not have have a langchain compatible LLM as a model attribute."
                    "To use default ValidMind LLM, do not set judge_llm/judge_embedding parameter, "
                    "ensure that you are connected to the ValidMind API and confirm ValidMind AI is enabled for your account."
                )
        if isinstance(judge_embeddings, FunctionModel) and judge_embeddings is not None:
            if isinstance(judge_embeddings.model, Embeddings):
                judge_embeddings = judge_embeddings.model
            else:
                raise ValueError(
                    "The ValidMind Functional model provided does not have have a langchain compatible embeddings model as a model attribute."
                    "To use default ValidMind LLM, do not set judge_embedding parameter, "
                    "ensure that you are connected to the ValidMind API and confirm ValidMind AI is enabled for your account."
                )

        if (isinstance(judge_llm, BaseChatModel) or judge_llm is None) and (
            isinstance(judge_embeddings, Embeddings) or judge_embeddings is None
        ):
            return judge_llm, judge_embeddings
        else:
            raise ValueError(
                "Provided Judge LLM/Embeddings are not Langchain compatible. Ensure the judge LLM/embedding provided are an instance of "
                "Langchain BaseChatModel and LangchainEmbeddings.  To use default ValidMind LLM, do not set judge_llm/judge_embedding parameter, "
                "ensure that you are connected to the ValidMind API and confirm ValidMind AI is enabled for your account."
            )

    # grab default values if not passed at run time
    global __judge_llm, __judge_embeddings
    if __judge_llm and __judge_embeddings:
        return __judge_llm, __judge_embeddings

    client, model = get_client_and_model()
    os.environ["OPENAI_API_BASE"] = str(client.base_url)

    __judge_llm = ChatOpenAI(api_key=client.api_key, model=model)
    __judge_embeddings = OpenAIEmbeddings(
        api_key=client.api_key, model=EMBEDDINGS_MODEL
    )

    return __judge_llm, __judge_embeddings


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
        client, model = get_client_and_model()
        # send an empty message with max_tokens=1 to "ping" the API
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ""}],
            max_tokens=1,
        )
        logger.debug(
            f"Received response from OpenAI: {response.choices[0].message.content}"
        )
        __ack = True
    except Exception as e:
        logger.debug(f"Failed to connect to OpenAI: {e}")
        __ack = False

    return __ack
