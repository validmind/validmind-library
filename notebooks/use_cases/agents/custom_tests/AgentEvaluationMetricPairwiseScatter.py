"""
Reusable pairwise metric scatter plot for Health Assistant agent evaluations.
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
        "score_column": "ToolCorrectness_score",
        "threshold": 0.7,
    },
    {
        "metric": "ArgumentCorrectness",
        "score_column": "ArgumentCorrectness_score",
        "threshold": 0.7,
    },
    {
        "metric": "AnswerRelevancy",
        "score_column": "AnswerRelevancy_score",
        "threshold": 0.8,
    },
    {
        "metric": "ContextualRelevancy",
        "score_column": "ContextualRelevancy_score",
        "threshold": 0.5,
    },
    {
        "metric": "Hallucination",
        "score_column": "Hallucination_score",
        "threshold": 0.5,
    },
    {
        "metric": "Bias",
        "score_column": "Bias_score",
        "threshold": 0.5,
    },
    {
        "metric": "Toxicity",
        "score_column": "GEval_Toxicity_score",
        "threshold": 0.5,
    },
]


def _build_metric_lookup() -> Dict[str, Dict[str, Any]]:
    """Build a lookup by metric name and score column."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for metric_config in DEFAULT_METRICS:
        lookup[metric_config["metric"]] = metric_config
        lookup[metric_config["score_column"]] = metric_config
    return lookup


def _resolve_metric(metric_or_column: str) -> Dict[str, Any]:
    """Resolve a metric name or score column into a metric configuration."""
    metric_lookup = _build_metric_lookup()
    if metric_or_column not in metric_lookup:
        available_values = sorted(metric_lookup.keys())
        raise ValueError(
            f"Unknown metric '{metric_or_column}'. "
            f"Use one of: {available_values}"
        )
    return metric_lookup[metric_or_column]


def _truncate_text(value: Any, max_length: int = 120) -> str:
    """Truncate long text fields for hover labels."""
    if value is None:
        return ""
    text = str(value).strip().replace("\n", " ")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


@tags("agent_evaluation", "visualization", "llm")
@tasks("data_description")
def AgentEvaluationMetricPairwiseScatter(
    dataset: VMDataset,
    x_metric: str = "ContextualRelevancy",
    y_metric: str = "Hallucination",
) -> go.Figure:
    """
    Create a reusable pairwise scatter plot for two agent evaluation metrics.

    This custom test is intended for targeted pairwise analysis rather than a full
    pairwise grid. It accepts two metric identifiers and plots one trace per point,
    allowing reviewers to inspect whether weaknesses tend to co-occur between two
    evaluation dimensions.

    Metric parameters may be passed either as friendly metric names such as
    ``ContextualRelevancy`` or as explicit score columns such as
    ``ContextualRelevancy_score``.

    Args:
        dataset: Scored evaluation dataset built from the trace dataframe.
        x_metric: Metric name or score column to plot on the x-axis.
        y_metric: Metric name or score column to plot on the y-axis.

    Returns:
        plotly.graph_objects.Figure: Pairwise scatter plot for the two selected metrics.
    """
    x_config = _resolve_metric(x_metric)
    y_config = _resolve_metric(y_metric)

    df = dataset.df.copy()
    x_column = x_config["score_column"]
    y_column = y_config["score_column"]

    missing_columns = [column for column in [x_column, y_column] if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Required score columns not found in dataset: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    plot_df = df[[x_column, y_column]].copy()
    plot_df[x_column] = pd.to_numeric(plot_df[x_column], errors="coerce")
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_column, y_column])

    if "id" in df.columns:
        plot_df["trace_id"] = df.loc[plot_df.index, "id"]
    else:
        plot_df["trace_id"] = plot_df.index.astype(str)

    if "input" in df.columns:
        plot_df["input_excerpt"] = df.loc[plot_df.index, "input"].apply(_truncate_text)
    else:
        plot_df["input_excerpt"] = ""

    fig = px.scatter(
        plot_df,
        x=x_column,
        y=y_column,
        hover_data={
            "trace_id": True,
            "input_excerpt": True,
            x_column: ":.3f",
            y_column: ":.3f",
        },
        title=f"{x_config['metric']} vs {y_config['metric']}",
        labels={
            x_column: x_config["metric"],
            y_column: y_config["metric"],
            "trace_id": "Trace ID",
            "input_excerpt": "Input",
        },
        template="plotly_white",
    )

    fig.add_vline(
        x=x_config["threshold"],
        line_dash="dash",
        line_color="gray",
        opacity=0.8,
    )
    fig.add_hline(
        y=y_config["threshold"],
        line_dash="dash",
        line_color="gray",
        opacity=0.8,
    )

    fig.update_traces(marker=dict(size=10, opacity=0.75))
    fig.update_layout(
        xaxis_title=x_config["metric"],
        yaxis_title=y_config["metric"],
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(range=[0, 1.05]),
        width=850,
        height=650,
        showlegend=False,
    )

    return fig
