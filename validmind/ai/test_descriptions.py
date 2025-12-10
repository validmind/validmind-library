# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from ..client_config import client_config
from ..logging import get_logger
from ..utils import NumpyEncoder, md_to_html, test_id_to_name
from ..vm_models.figure import Figure
from ..vm_models.result import ResultTable
from ..vm_models.result.pii_filter import (
    PIIDetectionMode,
    get_pii_detection_mode,
    scan_df,
)
from .utils import DescriptionFuture

__executor = ThreadPoolExecutor()

logger = get_logger(__name__)

# Try to import tiktoken once at module load
# Cache the result to avoid repeated import attempts
_TIKTOKEN_AVAILABLE = False
_TIKTOKEN_ENCODING = None

try:
    import tiktoken

    _TIKTOKEN_ENCODING = tiktoken.encoding_for_model("gpt-4o")
    _TIKTOKEN_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.debug(
        f"tiktoken unavailable, will use character-based token estimation: {e}"
    )


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


def _estimate_tokens_simple(text: str) -> int:
    """Estimate token count using character-based heuristic.

    Uses ~4 characters per token for English/JSON text.
    This is a fallback when tiktoken is unavailable.
    """
    return len(text) // 4


def _truncate_text_simple(text: str, max_tokens: int) -> str:
    """Truncate text using character-based estimation."""
    estimated_chars = max_tokens * 4
    if len(text) <= estimated_chars:
        return text

    # Keep first portion and last 100 tokens worth (~400 chars)
    # But ensure we don't take more than 25% for the tail
    last_chars = min(400, estimated_chars // 4)
    first_chars = estimated_chars - last_chars

    return text[:first_chars] + "...[truncated]" + text[-last_chars:]


def _truncate_summary(
    summary: Union[str, None], test_id: str, max_tokens: int = 100_000
):
    if summary is None or len(summary) < max_tokens:
        # since string itself is less than max_tokens, definitely small enough
        return summary

    if _TIKTOKEN_AVAILABLE:
        # Use tiktoken for accurate token counting
        summary_tokens = _TIKTOKEN_ENCODING.encode(summary)

        if len(summary_tokens) > max_tokens:
            logger.warning(
                f"Truncating {test_id} due to context length restrictions..."
                " Generated description may be inaccurate"
            )
            summary = (
                _TIKTOKEN_ENCODING.decode(summary_tokens[:max_tokens])
                + "...[truncated]"
                + _TIKTOKEN_ENCODING.decode(summary_tokens[-100:])
            )
    else:
        # Fallback to character-based estimation
        estimated_tokens = _estimate_tokens_simple(summary)

        if estimated_tokens > max_tokens:
            logger.warning(
                f"Truncating {test_id} (estimated) due to context length restrictions..."
                " Generated description may be inaccurate"
            )
            summary = _truncate_text_simple(summary, max_tokens)

    return summary


def generate_description(
    test_id: str,
    test_description: str,
    tables: List[ResultTable] = None,
    metric: Union[float, int] = None,
    figures: List[Figure] = None,
    title: Optional[str] = None,
    instructions: Optional[str] = None,
    additional_context: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
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
        if get_pii_detection_mode() in [
            PIIDetectionMode.TEST_DESCRIPTIONS,
            PIIDetectionMode.ALL,
        ]:
            for table in tables:
                scan_df(table.data)

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
            "figures": [figure._get_b64_url() for figure in figures or []],
            "additional_context": additional_context,
            "instructions": instructions,
            "params": params,
        }
    )["content"]


def background_generate_description(
    test_id: str,
    test_description: str,
    tables: List[ResultTable] = None,
    figures: List[Figure] = None,
    metric: Union[int, float] = None,
    title: Optional[str] = None,
    instructions: Optional[str] = None,
    additional_context: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
):
    def wrapped():
        try:
            return (
                generate_description(
                    test_id=test_id,
                    test_description=test_description,
                    tables=tables,
                    figures=figures,
                    metric=metric,
                    title=title,
                    instructions=instructions,
                    additional_context=additional_context,
                    params=params,
                ),
                True,
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

            return test_description, False

    return DescriptionFuture(__executor.submit(wrapped))


def get_result_description(
    test_id: str,
    test_description: str,
    tables: List[ResultTable] = None,
    figures: List[Figure] = None,
    metric: Union[int, float] = None,
    should_generate: bool = True,
    title: Optional[str] = None,
    instructions: Optional[str] = None,
    additional_context: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
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
        instructions (Optional[str]): Instructions for the LLM to generate the description.
        additional_context (Optional[str]): Additional context for the LLM to generate the description.
        params (Optional[Dict[str, Any]]): Test parameters used to customize test behavior.

    Returns:
        str: The description to be logged with the test results.
    """
    # Backwards compatibility: parameter instructions override environment variable
    env_instructions = _get_llm_global_context()
    # Parameter instructions take precedence and override environment variable
    _instructions = instructions if instructions is not None else env_instructions

    # Check the feature flag first, then the environment variable
    llm_descriptions_enabled = (
        client_config.can_generate_llm_test_descriptions()
        and os.getenv("VALIDMIND_LLM_DESCRIPTIONS_ENABLED", "1").lower()
        not in ["0", "false"]
    )

    if should_generate and (tables or figures) and llm_descriptions_enabled:
        # get description future and set it as the description in the metadata
        # this will lazily retrieved so it can run in the background in parallel
        description = background_generate_description(
            test_id=test_id,
            test_description=test_description,
            tables=tables,
            figures=figures,
            metric=metric,
            title=title,
            instructions=_instructions,
            additional_context=additional_context,
            params=params,
        )

    else:
        description = md_to_html(test_description, mathml=True)

    return description
