# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List

from validmind import tags, tasks
from validmind.ai.utils import get_client_and_model
from validmind.errors import MissingDependencyError
from validmind.tests.decorator import scorer
from validmind.vm_models.dataset import VMDataset

try:
    from deepeval import evaluate
    from deepeval.metrics import PlanAdherenceMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for PlanAdherence. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


@scorer()
@tags(
    "llm", "PlanAdherence", "deepeval", "agent_evaluation", "reasoning_layer", "agentic"
)
@tasks("llm")
def PlanAdherence(
    dataset: VMDataset,
    threshold: float = 0.7,
    input_column: str = "input",
    tools_called_column: str = "tools_called",
    actual_output_column: str = "actual_output",
    expected_output_column: str = "expected_output",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluates agent plan adherence using deepeval's PlanAdherenceMetric.

    This metric evaluates whether your agent follows its own plan during execution.
    Creating a good plan is only half the battle—an agent that deviates from its
    strategy mid-execution undermines its own reasoning.

    Args:
        dataset: Dataset containing the agent input, plan, and execution steps
        threshold: Minimum passing threshold (default: 0.7)
        input_column: Column name for the task input (default: "input")
        plan_column: Column name for the agent's plan (default: "plan")
        execution_steps_column: Column name for execution steps (default: "execution_steps")
        agent_output_column: Column name for agent output containing plan and steps (default: "agent_output")
        tools_called_column: Column name for tools called (default: "tools_called")
        strict_mode: If True, enforces a binary score (0 or 1)

    Returns:
        List[Dict[str, Any]] with keys "score" and "reason" for each row.

    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns exist in dataset
    missing_columns: List[str] = []
    if input_column not in dataset._df.columns:
        missing_columns.append(input_column)

    if tools_called_column not in dataset._df.columns:
        missing_columns.append(tools_called_column)

    if actual_output_column not in dataset._df.columns:
        missing_columns.append(actual_output_column)

    if expected_output_column not in dataset._df.columns:
        missing_columns.append(expected_output_column)

    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset. "
            f"Available columns: {dataset._df.columns.tolist()}"
        )

    _, model = get_client_and_model()

    metric = PlanAdherenceMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
        strict_mode=strict_mode,
        verbose_mode=False,
    )

    results: List[Dict[str, Any]] = []
    for _, row in dataset._df.iterrows():
        input_value = row[input_column]
        actual_output_value = row.get(actual_output_column, "")
        expected_output_value = row.get(expected_output_column, "")

        tools_called_value = row.get(tools_called_column, [])

        test_case = LLMTestCase(
            input=input_value,
            actual_output=actual_output_value,
            expected_output=expected_output_value,
            tools_called=tools_called_value,
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")
        results.append({"score": score, "reason": reason})

    return results
