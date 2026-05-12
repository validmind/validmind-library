# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Load bundled Health Assistant traces into DeepEval datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from deepeval.dataset import EvaluationDataset
    from deepeval.test_case import LLMTestCase, ToolCall

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    EvaluationDataset = None
    LLMTestCase = None
    ToolCall = None

current_path = Path(__file__).resolve().parent
dataset_file = current_path / "datasets" / "health_assistant.json"


def _to_tool_call(tool_trace: Dict[str, Any]) -> ToolCall:
    """Build a DeepEval tool call from one Health Assistant trace entry."""
    return ToolCall(
        name=tool_trace["name"],
        description=tool_trace.get("description"),
        reasoning=tool_trace.get("reasoning"),
        input_parameters=tool_trace.get("input_parameters"),
        output=tool_trace.get("output"),
    )


def _to_test_case(trace: Dict[str, Any]) -> LLMTestCase:
    """Build a DeepEval test case from one Health Assistant trace."""
    tools_called = [_to_tool_call(tool) for tool in trace.get("tools_called", [])]
    expected_tools = [_to_tool_call(tool) for tool in trace.get("expected_tools", [])]
    scenario = trace.get("scenario")
    tags = [scenario] if scenario else trace.get("tags")

    return LLMTestCase(
        name=trace.get("name"),
        tags=tags,
        input=trace["input"],
        actual_output=trace["actual_output"],
        expected_output=trace.get("expected_output"),
        context=trace.get("context"),
        retrieval_context=trace.get("retrieval_context"),
        tools_called=tools_called or None,
        expected_tools=expected_tools or None,
        additional_metadata=trace.get("additional_metadata"),
    )


def _load_traces(json_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load and validate Health Assistant traces from the bundled JSON file."""
    resolved_path = Path(json_path) if json_path is not None else dataset_file

    with open(resolved_path, encoding="utf-8") as data_file:
        traces = json.load(data_file)

    if not isinstance(traces, list):
        raise ValueError(
            "Expected a JSON array of traces at "
            f"{resolved_path}, got {type(traces).__name__}"
        )

    return traces


def load_deepeval_dataset(
    json_path: Optional[Union[str, Path]] = None,
) -> EvaluationDataset:
    """Load Health Assistant traces from JSON into a DeepEval evaluation dataset."""
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is required to load the Health Assistant dataset. "
            "Install it with: pip install deepeval"
        )

    traces = _load_traces(json_path=json_path)
    dataset = EvaluationDataset()
    for trace in traces:
        dataset.add_test_case(_to_test_case(trace))

    return dataset


def load_data(
    json_path: Optional[Union[str, Path]] = None,
) -> EvaluationDataset:
    """Load Health Assistant traces into a DeepEval evaluation dataset."""
    return load_deepeval_dataset(json_path=json_path)
