# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List

from validmind import tags, tasks
from validmind.ai.utils import get_client_and_model
from validmind.errors import MissingDependencyError
from validmind.tests.decorator import scorer
from validmind.vm_models.dataset import VMDataset

try:
    from deepeval import evaluate
    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for ToolCorrectness. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


@scorer()
@tags(
    "llm", "ToolCorrectness", "deepeval", "agent_evaluation", "action_layer", "agentic"
)
@tasks("llm")
def ToolCorrectness(
    dataset: VMDataset,
    threshold: float = 0.7,
    input_column: str = "input",
    expected_tools_called_column: str = "expected_tools",
    actual_tools_called_column: str = "tools_called",
) -> List[Dict[str, Any]]:
    """
    Evaluate the correctness of tool usage in LLM agent tasks using DeepEval's ToolCorrectnessMetric.

    This scorer checks if the agent invoked the expected tools for each task instance. It compares
    the list of tools the agent actually called to the reference (expected) tool calls on a per-row basis.
    The metric is suitable for scenarios where LLM agents use tools or external functions in their responses.

    If a `model` is supplied, it is used to predict/trace agent tool usage on the fly; otherwise, tool call columns
    must be prepopulated in the dataset.

    Args:
        dataset (VMDataset): Dataset containing task inputs, expected tool calls, and actual tool calls.
        threshold (float, optional): Passing threshold for tool match (default: 0.7).
        input_column (str, optional): Name of the column containing the LLM input prompt (default: "input").
        expected_tools_called_column (str, optional): Column with reference/expected tool calls.
        actual_tools_called_column (str, optional): Column with agent's actual tool calls.

    Returns:
        List[Dict[str, Any]]: List of dicts, one per dataset row, with:
            - "score" (float): Tool correctness score in [0,1]. 1.0 means all expected tools were called.
            - "reason" (str): Explanation or diagnostic from the DeepEval metric.

    Raises:
        ValueError: If any required columns are missing from the provided dataset.

    Example:
        >>> results = ToolCorrectness(dataset=my_data)
        >>> print(results[0]["score"])  # 1.0 if tools match, <1.0 if not

    Risks & Limitations:
        - Designed for table-formatted datasets with expected/actual tool call annotations.
        - Requires unambiguous tool call information for accurate evaluation.
        - May not fully support edge case tool calling formats or custom tracing logic.

    """
    from validmind.scorers.llm.deepeval import _convert_to_tool_call_list

    missing_columns: List[str] = []
    if input_column not in dataset._df.columns:
        missing_columns.append(input_column)
    if expected_tools_called_column not in dataset._df.columns:
        missing_columns.append(expected_tools_called_column)
    if actual_tools_called_column not in dataset._df.columns:
        missing_columns.append(actual_tools_called_column)
    if missing_columns:
        raise ValueError(
            f"ToolCorrectness with model requires columns {missing_columns}. "
            f"Available columns: {dataset._df.columns.tolist()}"
        )

    _, llm_model = get_client_and_model()
    results: List[Dict[str, Any]] = []

    for _, row in dataset._df.iterrows():
        expected_tools_value = row.get(expected_tools_called_column, [])
        expected_tools_list = _convert_to_tool_call_list(expected_tools_value)
        actual_tools_value = row.get(actual_tools_called_column, [])
        actual_tools_list = _convert_to_tool_call_list(actual_tools_value)

        metric = ToolCorrectnessMetric(
            threshold=threshold,
            model=llm_model,
        )

        test_case = LLMTestCase(
            input=row[input_column],
            expected_tools=expected_tools_list,
            tools_called=actual_tools_list,
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")
        results.append({"score": score, "reason": reason})
    return results
