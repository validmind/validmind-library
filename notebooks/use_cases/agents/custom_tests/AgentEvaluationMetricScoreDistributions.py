"""
Metric score distribution figure for Health Assistant agent evaluations.
"""

from __future__ import annotations

from typing import Dict, List

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
        "threshold": 0.7,
    },
    {
        "metric": "ArgumentCorrectness",
        "category": "Action Execution Quality",
        "score_column": "ArgumentCorrectness_score",
        "threshold": 0.7,
    },
    {
        "metric": "AnswerRelevancy",
        "category": "Response Quality And Grounding",
        "score_column": "AnswerRelevancy_score",
        "threshold": 0.8,
    },
    {
        "metric": "ContextualRelevancy",
        "category": "Response Quality And Grounding",
        "score_column": "ContextualRelevancy_score",
        "threshold": 0.5,
    },
    {
        "metric": "Hallucination",
        "category": "Response Quality And Grounding",
        "score_column": "Hallucination_score",
        "threshold": 0.5,
    },
    {
        "metric": "Bias",
        "category": "Safety And Harm Review",
        "score_column": "Bias_score",
        "threshold": 0.5,
    },
    {
        "metric": "Toxicity",
        "category": "Safety And Harm Review",
        "score_column": "GEval_Toxicity_score",
        "threshold": 0.5,
    },
]


def _build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Build a long-form dataframe for plotting metric distributions."""
    rows: List[Dict[str, Any]] = []
    for metric_config in DEFAULT_METRICS:
        score_column = metric_config["score_column"]
        if score_column not in df.columns:
            continue

        metric_scores = pd.to_numeric(df[score_column], errors="coerce")
        for row_idx, score in metric_scores.items():
            if pd.isna(score):
                continue
            rows.append(
                {
                    "metric": metric_config["metric"],
                    "score": float(score),
                    "trace_id": df.iloc[row_idx]["id"] if "id" in df.columns else row_idx,
                }
            )

    return pd.DataFrame(rows)


@tags("agent_evaluation", "visualization", "llm")
@tasks("data_description")
def AgentEvaluationMetricScoreDistributions(dataset: VMDataset) -> go.Figure:
    """
    Visualize score distributions across the scored agent evaluation metrics.

    This custom test creates a simple score distribution chart for the scored
    evaluation dataset. Each metric is shown as its own vertical box plot with all
    trace-level points displayed, making it easy to compare spread and outliers
    without adding extra visual layers.

    The test is intended to follow the metric summary and low-scoring trace review
    tables in the documentation.

    Args:
        dataset: Scored evaluation dataset built from the trace dataframe.

    Returns:
        plotly.graph_objects.Figure: Box plot for the available metric scores.
    """
    df = dataset.df.copy()
    plot_df = _build_long_format(df)

    if plot_df.empty:
        return go.Figure().update_layout(
            title="Metric Score Distributions",
            template="plotly_white",
        )

    metric_order = [metric["metric"] for metric in DEFAULT_METRICS if metric["score_column"] in df.columns]

    fig = px.box(
        plot_df,
        x="metric",
        y="score",
        points="all",
        category_orders={"metric": metric_order},
        hover_data={
            "trace_id": True,
            "metric": False,
            "score": ":.3f",
        },
        title="Metric Score Distributions",
        labels={
            "metric": "Metric",
            "score": "Score",
            "trace_id": "Trace ID",
        },
        template="plotly_white",
    )

    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        showlegend=False,
        width=1100,
        height=600,
    )
    fig.update_traces(
        jitter=0.25,
        pointpos=0,
        marker=dict(size=7, opacity=0.65),
        selector=dict(type="box"),
    )

    return fig
