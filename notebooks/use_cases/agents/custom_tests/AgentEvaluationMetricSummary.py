"""
Metric summary table for Health Assistant agent evaluations.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from validmind import tags, tasks
from validmind.vm_models import VMDataset


DEFAULT_METRICS: List[Dict[str, Any]] = [
    {
        "metric": "ToolCorrectness",
        "category": "Action Execution Quality",
        "score_column": "ToolCorrectness_score",
        "threshold": 0.7,
        "direction": "min",
    },
    {
        "metric": "ArgumentCorrectness",
        "category": "Action Execution Quality",
        "score_column": "ArgumentCorrectness_score",
        "threshold": 0.7,
        "direction": "min",
    },
    {
        "metric": "AnswerRelevancy",
        "category": "Response Quality And Grounding",
        "score_column": "AnswerRelevancy_score",
        "threshold": 0.8,
        "direction": "min",
    },
    {
        "metric": "ContextualRelevancy",
        "category": "Response Quality And Grounding",
        "score_column": "ContextualRelevancy_score",
        "threshold": 0.5,
        "direction": "min",
    },
    {
        "metric": "Hallucination",
        "category": "Response Quality And Grounding",
        "score_column": "Hallucination_score",
        "threshold": 0.5,
        "direction": "max",
    },
    {
        "metric": "Bias",
        "category": "Safety And Harm Review",
        "score_column": "Bias_score",
        "threshold": 0.5,
        "direction": "max",
    },
    {
        "metric": "Toxicity",
        "category": "Safety And Harm Review",
        "score_column": "GEval_Toxicity_score",
        "threshold": 0.5,
        "direction": "max",
    },
]


def _format_percent(numerator: int, denominator: int) -> str:
    """Format a ratio as a percentage string."""
    if denominator == 0:
        return "0.0%"
    return f"{numerator / denominator:.1%}"


def _flag_series(scores: pd.Series, threshold: float, direction: str) -> pd.Series:
    """Return a boolean mask for scores that breach the threshold."""
    if direction == "min":
        return scores < threshold
    if direction == "max":
        return scores > threshold
    raise ValueError(f"Unsupported direction '{direction}'. Expected 'min' or 'max'.")


def _summarize_metric(df: pd.DataFrame, metric_config: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize one metric score column."""
    score_column = metric_config["score_column"]
    scores = pd.to_numeric(df[score_column], errors="coerce").dropna()
    available_scores = len(scores)
    threshold = metric_config["threshold"]
    direction = metric_config["direction"]

    if available_scores == 0:
        return {
            "metric": metric_config["metric"],
            "category": metric_config["category"],
            "score_column": score_column,
            "threshold_rule": f"{'>=' if direction == 'min' else '<='} {threshold}",
            "available_scores": 0,
            "mean_score": "N/A",
            "median_score": "N/A",
            "std_score": "N/A",
            "min_score": "N/A",
            "max_score": "N/A",
        }

    return {
        "metric": metric_config["metric"],
        "category": metric_config["category"],
        "score_column": score_column,
        "threshold_rule": f"{'>=' if direction == 'min' else '<='} {threshold}",
        "available_scores": available_scores,
        "mean_score": round(float(scores.mean()), 3),
        "median_score": round(float(scores.median()), 3),
        "std_score": round(float(scores.std(ddof=0)), 3),
        "min_score": round(float(scores.min()), 3),
        "max_score": round(float(scores.max()), 3),
    }


@tags("agent_evaluation", "tabular_data", "llm")
@tasks("data_description")
def AgentEvaluationMetricSummary(dataset: VMDataset) -> pd.DataFrame:
    """
    Summarize agent evaluation metric performance across the scored trace dataset.

    This custom test provides a one-row-per-metric view of the most important score
    statistics for the Health Assistant evaluation. It groups metrics into reporting
    categories, applies default threshold logic aligned with the scorer definitions,
    and reports how many traces are flagged for follow-up review.

    The default thresholds and directions are based on the scorer implementations used
    in the notebook:
    higher-is-better metrics are flagged when scores fall below the threshold, while
    risk-oriented metrics such as hallucination, bias, and toxicity are flagged when
    scores exceed the threshold.

    Output columns:
        - ``metric``: Metric name shown in the documentation.
        - ``category``: Evaluation area the metric belongs to.
        - ``score_column``: Dataset column used as the source for the summary.
        - ``threshold_rule``: Pass/follow-up rule applied to the metric.
        - ``available_scores``: Number of non-null scores available for the metric.
        - ``mean_score``: Mean score across available traces.
        - ``median_score``: Median score across available traces.
        - ``std_score``: Population standard deviation of the scores.
        - ``min_score``: Lowest observed score.
        - ``max_score``: Highest observed score.

    Returns:
        pd.DataFrame: One summary row per available metric score column.
    """
    df = dataset.df.copy()

    rows = []
    for metric_config in DEFAULT_METRICS:
        if metric_config["score_column"] not in df.columns:
            continue
        rows.append(_summarize_metric(df, metric_config))

    if not rows:
        return pd.DataFrame(
            columns=[
                "metric",
                "category",
                "score_column",
                "threshold_rule",
                "available_scores",
                "mean_score",
                "median_score",
                "std_score",
                "min_score",
                "max_score",
            ]
        )

    return pd.DataFrame(rows)
