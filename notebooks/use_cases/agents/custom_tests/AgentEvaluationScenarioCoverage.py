"""
Scenario and cohort coverage tables for Health Assistant agent evaluations.
"""

from __future__ import annotations

from collections import Counter
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


def _extract_metadata(dataset: VMDataset) -> Dict[str, List[Any]]:
    """Extract optional metadata from LLMAgentDataset test cases when available."""
    metadata: Dict[str, List[Any]] = {
        "scenarios": [],
        "age_bands": [],
        "regions": [],
        "plan_types": [],
        "plan_tiers": [],
    }

    test_cases = getattr(dataset, "test_cases", None) or []
    for test_case in test_cases:
        tags_value = getattr(test_case, "tags", None) or []
        if tags_value:
            metadata["scenarios"].append(tags_value[0])

        additional_metadata = getattr(test_case, "additional_metadata", None) or {}
        participant_metadata = additional_metadata.get("participant", {})
        cohort_metadata = participant_metadata.get("cohort", {})

        metadata["age_bands"].append(cohort_metadata.get("age_band"))
        metadata["regions"].append(cohort_metadata.get("region"))
        metadata["plan_types"].append(cohort_metadata.get("plan_type"))
        metadata["plan_tiers"].append(cohort_metadata.get("plan_tier"))

    return metadata


def _build_scenario_table(metadata: Dict[str, List[Any]], total_traces: int) -> pd.DataFrame:
    """Build one row per scenario when scenario metadata is available."""
    scenario_counter = Counter(
        str(scenario) for scenario in metadata["scenarios"] if _is_present(scenario)
    )
    if not scenario_counter:
        return pd.DataFrame(columns=["scenario", "trace_count", "pct_of_dataset"])

    rows = [
        {
            "scenario": scenario,
            "trace_count": trace_count,
            "pct_of_dataset": round(trace_count / total_traces, 4) if total_traces else 0.0,
        }
        for scenario, trace_count in scenario_counter.most_common()
    ]
    return pd.DataFrame(rows)


def _build_participant_coverage_table(metadata: Dict[str, List[Any]]) -> pd.DataFrame:
    """Build a compact participant coverage table."""
    cohort_specs = [
        ("age_bands", "age_band"),
        ("regions", "region"),
        ("plan_types", "plan_type"),
        ("plan_tiers", "plan_tier"),
    ]

    rows = []
    for field_name, label in cohort_specs:
        values = sorted(
            {str(value) for value in metadata[field_name] if _is_present(value)}
        )
        if not values:
            continue
        rows.append(
            {
                "dimension": label,
                "unique_values": len(values),
                "values": ", ".join(values),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["dimension", "unique_values", "values"])

    return pd.DataFrame(rows)


@tags("agent_evaluation", "tabular_data", "llm")
@tasks("data_description")
def AgentEvaluationScenarioCoverage(dataset: VMDataset) -> Dict[str, pd.DataFrame]:
    """
    Summarize scenario and participant cohort coverage in the agent evaluation set.

    This custom test complements ``AgentEvaluationDatasetOverview`` by focusing on
    the composition of the evaluation set rather than the overall dataset shape.
    It summarizes how traces are distributed across scenarios and which participant
    cohort attributes are represented in the evaluation traces.

    Returns:
        Dict[str, pd.DataFrame]: Two tables:
            - ``Scenario Coverage``: one row per scenario with trace counts and share of
              the dataset.
            - ``Participant Coverage``: one row per participant cohort dimension with
              unique values represented in the evaluation set.
    """
    metadata = _extract_metadata(dataset)

    return {
        "Scenario Coverage": _build_scenario_table(metadata, total_traces=len(dataset.df)),
        "Participant Coverage": _build_participant_coverage_table(metadata),
    }
