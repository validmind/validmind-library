# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os
from urllib.parse import urljoin

from openai import AzureOpenAI, Client, OpenAI

from ..logging import get_logger
from ..utils import md_to_html

logger = get_logger(__name__)


__client = None
__model = None
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
        if isinstance(self._future, str):
            description = self._future
        else:
            # This will block until the future is completed
            description = self._future.result()

        return md_to_html(description, mathml=True)


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
        try:
            # TODO: fix circular import
            from ..api_client import get_ai_key, get_api_host

            response = get_ai_key()
            __client = Client(
                base_url=(
                    # TODO: improve this to be a bit more dynamic
                    "http://localhost:4000/genai"
                    if "localhost" in get_api_host()
                    else urljoin(get_api_host(), "/genai")
                ),
                api_key=response["key"],
            )
            __model = "gpt-4o"  # TODO: backend should tell us which model to use
            logger.debug(f"Using ValidMind {__model} for generating descriptions")
        except Exception as e:
            logger.debug(f"Failed to get API key: {e}")
            raise ValueError(
                "OPENAI_API_KEY, AZURE_OPENAI_KEY must be set, or your account "
                "must be setup to use ValidMind's LLM in order to use LLM features"
            )

    return __client, __model


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
