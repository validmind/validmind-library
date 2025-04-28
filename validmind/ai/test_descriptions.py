# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import tiktoken

from ..client_config import client_config
from ..logging import get_logger
from ..utils import NumpyEncoder, md_to_html, test_id_to_name
from ..vm_models.figure import Figure
from ..vm_models.result import ResultTable
from .utils import DescriptionFuture

__executor = ThreadPoolExecutor()

logger = get_logger(__name__)


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


def _truncate_summary(
    summary: Union[str, None], test_id: str, max_tokens: int = 100_000
):
    if summary is None or len(summary) < max_tokens:
        # since string itself is less than max_tokens, definitely small enough
        return summary

    # TODO: better context length handling
    encoding = tiktoken.encoding_for_model("gpt-4o")
    summary_tokens = encoding.encode(summary)

    if len(summary_tokens) > max_tokens:
        logger.warning(
            f"Truncating {test_id} due to context length restrictions..."
            " Generated description may be innacurate"
        )
        summary = (
            encoding.decode(summary_tokens[:max_tokens])
            + "...[truncated]"
            + encoding.decode(summary_tokens[-100:])
        )

    return summary


def generate_description(
    test_id: str,
    test_description: str,
    tables: List[ResultTable] = None,
    metric: Union[float, int] = None,
    figures: List[Figure] = None,
    title: Optional[str] = None,
):
    """Generate the description for the test results."""
    from validmind.api_client import generate_test_result_description

    if not tables and not figures and not metric:
        raise ValueError(
            "No tables, unit metric or figures provided - cannot generate description"
        )

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

    return generate_test_result_description(
        {
            "test_name": test_name,
            "test_description": test_description,
            "title": title,
            "summary": _truncate_summary(summary, test_id),
            "figures": [
                figure._get_b64_url() for figure in ([] if tables else figures)
            ],
            "context": _get_llm_global_context(),
        }
    )["content"]


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
            if "maximum context length" in str(e):
                logger.warning(
                    f"Test result {test_id} is too large to generate a description"
                )
            elif "Too many images" in str(e):
                logger.warning(
                    f"Test result {test_id} has too many figures to generate a description"
                )
            else:
                logger.warning(f"Failed to generate description for {test_id}: {e}")
            logger.warning(f"Using default description for {test_id}")

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
    """Get the metadata dictionary for a test or metric result.

    Generates an LLM interpretation of the test results or uses the default
    description and returns a metadata object that can be logged with the test results.

    By default, the description is generated by an LLM that will interpret the test
    results and provide a human-readable description. If the tables or figures are
    not provided, or the `VALIDMIND_LLM_DESCRIPTIONS_ENABLED` environment variable is
    set to `0` or `false` or no LLM has been configured, the default description will
    be used as the test result description.

    Note: Either the tables or figures must be provided to generate the description.

    Args:
        test_id (str): The test ID.
        test_description (str): The default description for the test.
        tables (Any): The test tables or results to interpret.
        figures (List[Figure]): The figures to attach to the test suite result.
        metric (Union[int, float]): Unit metrics attached to the test result.
        should_generate (bool): Whether to generate the description or not. Defaults to True.

    Returns:
        str: The description to be logged with the test results.
    """
    # Check the feature flag first, then the environment variable
    llm_descriptions_enabled = (
        client_config.can_generate_llm_test_descriptions()
        and os.getenv("VALIDMIND_LLM_DESCRIPTIONS_ENABLED", "1").lower()
        not in ["0", "false"]
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
