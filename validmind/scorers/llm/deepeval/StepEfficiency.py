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
    from deepeval.metrics import StepEfficiencyMetric
    from deepeval.test_case import LLMTestCase, ToolCall
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for StepEfficiency. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


def _validate_columns(
    dataset: VMDataset,
    input_column: str,
    actual_output_column: str,
) -> None:
    """Raise ValueError if required columns are missing from the dataset."""
    missing_columns: List[str] = []
    if input_column not in dataset._df.columns:
        missing_columns.append(input_column)
    if actual_output_column not in dataset._df.columns:
        missing_columns.append(actual_output_column)
    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset. "
            f"Available columns: {dataset._df.columns.tolist()}"
        )


def _normalize_tools_called(tools_called_value: Any) -> List[Any]:
    """Convert tools_called to a list of ToolCall if needed."""
    if isinstance(tools_called_value, list) and all(
        isinstance(tool, ToolCall) for tool in tools_called_value
    ):
        return tools_called_value
    from validmind.scorers.llm.deepeval import _convert_to_tool_call_list

    return _convert_to_tool_call_list(tools_called_value)


def _build_trace_dict(
    trace_dict: Any,
    input_value: Any,
    actual_output_value: Any,
) -> Dict[str, Any]:
    """Ensure trace_dict is a dict with 'input' and 'output' for task extraction."""
    if not isinstance(trace_dict, dict):
        trace_dict = {}
    if "input" not in trace_dict:
        trace_dict = {**trace_dict, "input": input_value}
    if "output" not in trace_dict:
        trace_dict = {**trace_dict, "output": actual_output_value}
    return trace_dict


def _evaluate_single_case(
    test_case: LLMTestCase,
    metric: StepEfficiencyMetric,
) -> Dict[str, Any]:
    """Run StepEfficiencyMetric on one test case; return score/reason or error result."""
    try:
        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")
        return {"score": score, "reason": reason}
    except (UnboundLocalError, AttributeError, KeyError) as e:
        error_msg = str(e)
        if "prompt" in error_msg or "referenced before assignment" in error_msg:
            return {
                "score": 0.0,
                "reason": (
                    f"StepEfficiency evaluation failed: The agent trace may not contain "
                    f"sufficient execution steps for analysis. StepEfficiencyMetric requires "
                    f"a complete execution trace with step-by-step actions. "
                    f"Original error: {error_msg}"
                ),
            }
        raise
    except Exception as e:
        return {
            "score": 0.0,
            "reason": (
                f"StepEfficiency evaluation failed: {str(e)}. "
                f"This metric requires a properly structured agent execution trace."
            ),
        }


@scorer()
@tags("llm", "deepeval", "agent_evaluation", "action_layer", "agentic")
@tasks("llm")
def StepEfficiency(
    dataset: VMDataset,
    threshold: float = 0.5,
    input_column: str = "input",
    actual_output_column: str = "actual_output",
    agent_output_column: str = "agent_output",
    tools_called_column: str = "tools_called",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluates agent step efficiency using deepeval's StepEfficiencyMetric.

    This metric evaluates whether the agent avoids unnecessary or redundant steps
    in completing the given task. It analyzes the agent's full execution trace
    to assess the efficiency of the execution steps.

    Note: StepEfficiencyMetric requires a complete execution trace with step-by-step
    actions. If the trace structure is incomplete or doesn't contain sufficient
    execution steps, the evaluation may fail and return a score of 0.0 with an
    explanatory reason.

    Args:
        dataset: Dataset containing the agent input and execution trace
        threshold: Minimum passing threshold (default: 0.5)
        input_column: Column name for the task input (default: "input")
        actual_output_column: Column name for the agent's final output (default: "actual_output")
        agent_output_column: Column name for agent output containing trace (default: "agent_output")
        tools_called_column: Column name for tools called by the agent (default: "tools_called")
        strict_mode: If True, enforces a binary score (0 or 1)

    Returns:
        List[Dict[str, Any]] with keys "score" and "reason" for each row.
        If evaluation fails due to incomplete trace structure, returns score 0.0
        with an explanatory reason message.

    Raises:
        ValueError: If required columns are missing
    """
    _validate_columns(dataset, input_column, actual_output_column)

    _, model = get_client_and_model()
    metric = StepEfficiencyMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
        strict_mode=strict_mode,
        verbose_mode=False,
    )

    results: List[Dict[str, Any]] = []
    for _, row in dataset._df.iterrows():
        input_value = row[input_column]
        actual_output_value = row[actual_output_column]
        tools_called_value = _normalize_tools_called(row.get(tools_called_column, []))
        trace_dict = _build_trace_dict(
            row.get(agent_output_column, {}),
            input_value,
            actual_output_value,
        )
        test_case = LLMTestCase(
            input=input_value,
            actual_output=actual_output_value,
            tools_called=tools_called_value,
            _trace_dict=trace_dict,
        )
        results.append(_evaluate_single_case(test_case, metric))

    return results
