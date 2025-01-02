# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

from jinja2 import Template

from ..client_config import client_config
from ..logging import get_logger
from ..utils import NumpyEncoder, md_to_html, test_id_to_name
from ..vm_models.figure import Figure
from ..vm_models.result import ResultTable
from .utils import DescriptionFuture, get_client_and_model

__executor = ThreadPoolExecutor()
__prompt = None

logger = get_logger(__name__)


def _load_prompt():
    global __prompt

    if not __prompt:
        folder_path = os.path.join(os.path.dirname(__file__), "test_result_description")
        with open(os.path.join(folder_path, "system.jinja"), "r") as f:
            system_prompt = f.read()
        with open(os.path.join(folder_path, "user.jinja"), "r") as f:
            user_prompt = f.read()

        __prompt = (Template(system_prompt), Template(user_prompt))

    return __prompt


def prompt_to_message(role, prompt):
    if "[[IMAGE:" not in prompt:
        return {"role": role, "content": prompt}

    content = []

    # Regex pattern to find [[IMAGE:<b64-data>]] markers
    pattern = re.compile(r"\[\[IMAGE:(.*?)\]\]", re.DOTALL)

    last_index = 0
    for match in pattern.finditer(prompt):
        # Text before the image marker
        start, end = match.span()
        if start > last_index:
            content.append({"type": "text", "text": prompt[last_index:start]})

        content.append({"type": "image_url", "image_url": {"url": match.group(1)}})

        last_index = end

    # Text after the last image
    if last_index < len(prompt):
        content.append({"type": "text", "text": prompt[last_index:]})

    return {"role": role, "content": content}


def _get_llm_global_context():

    # Get the context from the environment variable
    context = os.getenv("VALIDMIND_LLM_DESCRIPTIONS_CONTEXT", "")

    # Check if context should be used (similar to descriptions enabled pattern)
    context_enabled = os.getenv(
        "VALIDMIND_LLM_DESCRIPTIONS_CONTEXT_ENABLED", "1"
    ) not in [
        "0",
        "false",
    ]

    # Only use context if it's enabled and not empty
    return context if context_enabled and context else None


def generate_description(
    test_id: str,
    test_description: str,
    tables: List[ResultTable] = None,
    metric: Union[float, int] = None,
    figures: List[Figure] = None,
    title: Optional[str] = None,
):
    """Generate the description for the test results"""
    if not tables and not figures and not metric:
        raise ValueError(
            "No tables, unit metric or figures provided - cannot generate description"
        )

    client, model = get_client_and_model()

    # get last part of test id
    test_name = title or test_id.split(".")[-1]

    if metric is not None:
        tables = [] if not tables else tables
        tables.append(
            ResultTable(
                data=[
                    {"Metric": test_id_to_name(test_id), "Value": metric},
                ],
            )
        )

    if tables:
        summary = "\n---\n".join(
            [
                json.dumps(table.serialize(), cls=NumpyEncoder, separators=(",", ":"))
                for table in tables
            ]
        )
    else:
        summary = None

    context = _get_llm_global_context()

    input_data = {
        "test_name": test_name,
        "test_description": test_description,
        "title": title,
        "summary": summary,
        "figures": [figure._get_b64_url() for figure in ([] if tables else figures)],
        "context": context,
    }
    system, user = _load_prompt()

    messages = [
        prompt_to_message("system", system.render(input_data)),
        prompt_to_message("user", user.render(input_data)),
    ]
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=messages,
    )

    return response.choices[0].message.content


def background_generate_description(
    test_id: str,
    test_description: str,
    tables: List[ResultTable] = None,
    figures: List[Figure] = None,
    metric: Union[int, float] = None,
    title: Optional[str] = None,
):
    def wrapped():
        try:
            return generate_description(
                test_id=test_id,
                test_description=test_description,
                tables=tables,
                figures=figures,
                metric=metric,
                title=title,
            )
        except Exception as e:
            logger.error(f"Failed to generate description: {e}")

            return test_description

    return DescriptionFuture(__executor.submit(wrapped))


def get_result_description(
    test_id: str,
    test_description: str,
    tables: List[ResultTable] = None,
    figures: List[Figure] = None,
    metric: Union[int, float] = None,
    should_generate: bool = True,
    title: Optional[str] = None,
):
    """Get Metadata Dictionary for a Test or Metric Result

    Generates an LLM interpretation of the test results or uses the default
    description and returns a metadata object that can be logged with the test results.

    By default, the description is generated by an LLM that will interpret the test
    results and provide a human-readable description. If the tables or figures are
    not provided, or the `VALIDMIND_LLM_DESCRIPTIONS_ENABLED` environment variable is
    set to `0` or `false` or no LLM has been configured, the default description will
    be used as the test result description.

    Note: Either the tables or figures must be provided to generate the description.

    Args:
        test_id (str): The test ID
        test_description (str): The default description for the test
        tables (Any): The test tables or results to interpret
        figures (List[Figure]): The figures to attach to the test suite result
        metric (Union[int, float]): Unit metrics attached to the test result
        should_generate (bool): Whether to generate the description or not (Default: True)

    Returns:
        str: The description to be logged with the test results
    """
    # Check the feature flag first, then the environment variable
    llm_descriptions_enabled = (
        client_config.can_generate_llm_test_descriptions()
        and os.getenv("VALIDMIND_LLM_DESCRIPTIONS_ENABLED", "1") not in ["0", "false"]
    )

    # TODO: fix circular import
    from validmind.ai.utils import is_configured

    if (
        should_generate
        and (tables or figures)
        and llm_descriptions_enabled
        and is_configured()
    ):
        # get description future and set it as the description in the metadata
        # this will lazily retrieved so it can run in the background in parallel
        description = background_generate_description(
            test_id=test_id,
            test_description=test_description,
            tables=tables,
            figures=figures,
            metric=metric,
            title=title,
        )

    else:
        description = md_to_html(test_description, mathml=True)

    return description
