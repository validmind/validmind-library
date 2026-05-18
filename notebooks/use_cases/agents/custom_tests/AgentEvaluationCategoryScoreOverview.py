"""
Category-level score overview figure for Health Assistant agent evaluations.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from validmind import tags, tasks
from validmind.vm_models import VMDataset


DEFAULT_METRICS: List[Dict[str, Any]] = [
    {
        "metric": "ToolCorrectness",
        "category": "Action Execution Quality",
        "score_column": "ToolCorrectness_score",
        "direction": "min",
    },
    {
        "metric": "ArgumentCorrectness",
        "category": "Action Execution Quality",
        "score_column": "ArgumentCorrectness_score",
        "direction": "min",
    },
    {
        "metric": "AnswerRelevancy",
        "category": "Response Quality And Grounding",
        "score_column": "AnswerRelevancy_score",
        "direction": "min",
    },
    {
        "metric": "ContextualRelevancy",
        "category": "Response Quality And Grounding",
        "score_column": "ContextualRelevancy_score",
        "direction": "min",
    },
    {
        "metric": "Hallucination",
        "category": "Response Quality And Grounding",
        "score_column": "Hallucination_score",
        "direction": "max",
    },
    {
        "metric": "Bias",
        "category": "Safety And Harm Review",
        "score_column": "Bias_score",
        "direction": "max",
    },
    {
        "metric": "Toxicity",
        "category": "Safety And Harm Review",
        "score_column": "GEval_Toxicity_score",
        "direction": "max",
    },
]


def _align_score(score: float, direction: str) -> float:
    """Convert metric scores to a common higher-is-better scale."""
    if direction == "min":
        return score
    if direction == "max":
        return 1.0 - score
    raise ValueError(f"Unsupported direction '{direction}'. Expected 'min' or 'max'.")


def _build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate aligned metric scores into category-level summaries."""
    metric_rows: List[Dict[str, Any]] = []

    for metric_config in DEFAULT_METRICS:
        score_column = metric_config["score_column"]
        if score_column not in df.columns:
            continue

        scores = pd.to_numeric(df[score_column], errors="coerce").dropna()
        if scores.empty:
            continue

        aligned_scores = scores.apply(
            lambda score: _align_score(float(score), metric_config["direction"])
        )
        metric_rows.append(
            {
                "metric": metric_config["metric"],
                "category": metric_config["category"],
                "aligned_mean_score": float(aligned_scores.mean()),
            }
        )

    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "category_mean_score",
                "metric_count",
                "metrics_included",
            ]
        )

    category_df = (
        metric_df.groupby("category", as_index=False)
        .agg(
            category_mean_score=("aligned_mean_score", "mean"),
            metric_count=("metric", "count"),
            metrics_included=("metric", lambda metric_values: ", ".join(metric_values)),
        )
        .sort_values("category_mean_score", ascending=False)
    )
    category_df["category_mean_score"] = category_df["category_mean_score"].round(3)
    return category_df


@tags("agent_evaluation", "visualization", "llm")
@tasks("data_description")
def AgentEvaluationCategoryScoreOverview(dataset: VMDataset) -> go.Figure:
    """
    Visualize evaluation performance at the category level using aligned scores.

    This custom test provides a higher-level companion to the metric score
    distribution figure. It aggregates metrics into the main evaluation areas:
    action execution quality, response quality and grounding, and safety and harm
    review.

    Because some metrics are higher-is-better and others are lower-is-better, the
    figure first converts all metric means to a common aligned score where higher is
    better. Risk-oriented metrics such as hallucination, bias, and toxicity are
    transformed as ``1 - score`` before category averages are computed.

    Args:
        dataset: Scored evaluation dataset built from the trace dataframe.

    Returns:
        plotly.graph_objects.Figure: Category-level aligned score bar chart.
    """
    category_df = _build_category_summary(dataset.df.copy())

    if category_df.empty:
        return go.Figure().update_layout(
            title="Category-Level Evaluation Overview",
            template="plotly_white",
        )

    fig = px.bar(
        category_df,
        x="category",
        y="category_mean_score",
        color="category",
        text="category_mean_score",
        hover_data={
            "metric_count": True,
            "metrics_included": True,
            "category": False,
            "category_mean_score": ":.3f",
        },
        title="Category-Level Evaluation Overview",
        labels={
            "category": "Evaluation Area",
            "category_mean_score": "Aligned Mean Score",
            "metric_count": "Metrics Included",
            "metrics_included": "Metric Set",
        },
        template="plotly_white",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Evaluation Area",
        yaxis_title="Aligned Mean Score",
        yaxis=dict(range=[0, 1.05]),
        showlegend=False,
        width=950,
        height=550,
    )

    return fig
