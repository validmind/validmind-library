"""
Dataset overview tables for Health Assistant agent evaluations.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd
from validmind import tags, tasks
from validmind.vm_models import VMDataset


def _is_present(value: Any) -> bool:
    """Return True when the value should count as populated."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _safe_unique_count(values: Iterable[Any]) -> int:
    """Count unique populated values."""
    cleaned_values = []
    for value in values:
        if not _is_present(value):
            continue
        cleaned_values.append(str(value))
    return len(set(cleaned_values))


def _format_percent(numerator: int, denominator: int) -> str:
    """Format a ratio as a percentage string."""
    if denominator == 0:
        return "0.0%"
    return f"{numerator / denominator:.1%}"


def _build_overview_table(
    df: pd.DataFrame, metadata: Dict[str, List[Any]]
) -> pd.DataFrame:
    """Build a compact high-level overview table with dimensions as rows."""
    total_traces = len(df)
    timestamp_values = sorted(
        str(timestamp)
        for timestamp in metadata["timestamps"]
        if _is_present(timestamp)
    )
    model_versions = sorted(
        {
            str(model_version)
            for model_version in metadata["model_versions"]
            if _is_present(model_version)
        }
    )

    expected_intents = metadata["expected_intents"]
    actual_intents = metadata["actual_intents"]
    comparable_intents = [
        (expected_intent, actual_intent)
        for expected_intent, actual_intent in zip(expected_intents, actual_intents)
        if _is_present(expected_intent) and _is_present(actual_intent)
    ]
    intent_agreement_rate = (
        _format_percent(
            sum(
                expected_intent == actual_intent
                for expected_intent, actual_intent in comparable_intents
            ),
            len(comparable_intents),
        )
        if comparable_intents
        else "N/A"
    )

    expected_scopes = metadata["expected_scopes"]
    actual_scopes = metadata["actual_scopes"]
    comparable_scopes = [
        (expected_scope, actual_scope)
        for expected_scope, actual_scope in zip(expected_scopes, actual_scopes)
        if _is_present(expected_scope) and _is_present(actual_scope)
    ]
    scope_agreement_rate = (
        _format_percent(
            sum(
                expected_scope == actual_scope
                for expected_scope, actual_scope in comparable_scopes
            ),
            len(comparable_scopes),
        )
        if comparable_scopes
        else "N/A"
    )

    rows = [
        {"dimension": "Total traces", "value": total_traces},
        {
            "dimension": "Unique sessions",
            "value": _safe_unique_count(metadata["sessions"]) or "N/A",
        },
        {
            "dimension": "Scenario count",
            "value": _safe_unique_count(metadata["scenarios"]) or "N/A",
        },
        {
            "dimension": "Model versions",
            "value": ", ".join(model_versions) if model_versions else "N/A",
        },
        {
            "dimension": "Evaluation start",
            "value": timestamp_values[0] if timestamp_values else "N/A",
        },
        {
            "dimension": "Evaluation end",
            "value": timestamp_values[-1] if timestamp_values else "N/A",
        },
        {"dimension": "Intent agreement rate", "value": intent_agreement_rate},
        {"dimension": "Scope agreement rate", "value": scope_agreement_rate},
    ]

    return pd.DataFrame(rows)


def _extract_metadata(dataset: VMDataset) -> Dict[str, List[Any]]:
    """Extract optional metadata from LLMAgentDataset test cases when available."""
    metadata: Dict[str, List[Any]] = {
        "trace_names": [],
        "sessions": [],
        "timestamps": [],
        "model_versions": [],
        "scenarios": [],
        "expected_intents": [],
        "actual_intents": [],
        "expected_scopes": [],
        "actual_scopes": [],
        "age_bands": [],
        "regions": [],
        "plan_types": [],
        "plan_tiers": [],
    }

    test_cases = getattr(dataset, "test_cases", None) or []
    for test_case in test_cases:
        metadata["trace_names"].append(getattr(test_case, "name", None))

        tags_value = getattr(test_case, "tags", None) or []
        if tags_value:
            metadata["scenarios"].append(tags_value[0])

        additional_metadata = getattr(test_case, "additional_metadata", None) or {}
        supervisor_metadata = additional_metadata.get("supervisor", {})
        participant_metadata = additional_metadata.get("participant", {})
        cohort_metadata = participant_metadata.get("cohort", {})

        metadata["sessions"].append(additional_metadata.get("session_id"))
        metadata["timestamps"].append(additional_metadata.get("timestamp"))
        metadata["model_versions"].append(additional_metadata.get("model_version"))
        metadata["expected_intents"].append(supervisor_metadata.get("expected_intent"))
        metadata["actual_intents"].append(supervisor_metadata.get("actual_intent"))
        metadata["expected_scopes"].append(supervisor_metadata.get("expected_scope"))
        metadata["actual_scopes"].append(supervisor_metadata.get("actual_scope"))
        metadata["age_bands"].append(cohort_metadata.get("age_band"))
        metadata["regions"].append(cohort_metadata.get("region"))
        metadata["plan_types"].append(cohort_metadata.get("plan_type"))
        metadata["plan_tiers"].append(cohort_metadata.get("plan_tier"))

    return metadata


@tags("agent_evaluation", "tabular_data", "llm")
@tasks("data_description")
def AgentEvaluationDatasetOverview(dataset: VMDataset) -> pd.DataFrame:
    """
    Summarize the evaluated agent trace dataset before reviewing metric scores.

    This custom test creates a compact overview of the evaluation dataset so reviewers
    can quickly understand the size and scope of the evaluation before reviewing
    scenarios, cohort coverage, or metric performance in more detail.

    The output is intentionally limited to a single high-level table with dimensions
    as rows and values in a second column. When the input is an ``LLMAgentDataset``,
    the test reads optional metadata from the underlying DeepEval test cases to
    summarize sessions, scenarios, routing agreement, model versions, and evaluation
    dates.

    Output columns:
        - ``dimension``: Name of the high-level dataset property being summarized.
        - ``value``: The summarized value for that property.

    Returns:
        pd.DataFrame: A single high-level dataset overview table with dimensions as rows.
    """
    df = dataset.df.copy()
    metadata = _extract_metadata(dataset)

    return _build_overview_table(df, metadata)
