"""
ValidMind test: manual weighted ranking and/or multi-preset scorecards (+ diagnostics).

Preset scorecard path uses :func:`~src.utils.experiments.metric_presets.render_leaderboard_views`
and :func:`~src.utils.experiments.metric_presets.merge_preset_leaderboards`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from validmind import tags, tasks
from validmind.vm_models import VMDataset

from src.utils.experiments.metric_presets import (
    diagnostics_summary,
    merge_preset_leaderboards,
    render_leaderboard_views,
)


@tags("tabular_data")
@tasks("data_description")
def MyTest(
    dataset: VMDataset,
    langfuse_dataset_name: Optional[str] = None,
    preset_names: Optional[list[str]] = None,
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    """
    Rank LLMs using preset scorecards with optional diagnostics summary.

    This function creates composite ranking columns for LLMs using named presets defined in
    ``config/metric_presets.yaml``. By default, it merges scorecards for ``quality_score`` and
    ``value_score`` (display columns ``Quality Score``, ``Value Score``), but alternative
    or additional presets may be specified via ``preset_names``.

    For each preset, a weighted composite score is calculated using the preset's metric weights
    and normalization rules. Higher scores indicate better models per preset's philosophy.
    The resulting per-preset leaderboards are aligned side by side for comparison.

    If ``include_diagnostics`` is True, descriptive metrics such as token counts and latency
    are aggregated per LLM and appended as additional columns.

    Note: Manual weighted scoring (passing explicit ``metric_weights``) is not currently
    supported in this interface; use ``rank_llms_by_weighted_score`` directly for that workflow.

    Args:
        dataset: VMDataset object containing a dataframe (``dataset.df``) with LLM experiment
            metrics, typically one row per LLM.
        langfuse_dataset_name: Optional; name of a dataset subset to filter/aggregate by.
        preset_names: Optional; list of preset keys to use for scorecard construction. If not
            specified, uses ["quality_score", "value_score"].
        include_diagnostics: If True (default), appends per-LLM means of diagnostic columns.

    Returns:
        A pandas DataFrame with one row per LLM, containing the composite scores for each preset
        (e.g., "Quality Score", "Value Score"), and, if requested, appended diagnostic summary columns.

        Score columns are named per preset (not always "Weighted Score"); higher is better.
        Diagnostic columns are descriptive and do not affect ranking unless included in a preset.
        Sort order is not guaranteed.
    """
    df = dataset.df
    names = (
        preset_names if preset_names is not None else ["quality_score", "value_score"]
    )

    views = render_leaderboard_views(
        df,
        langfuse_dataset=langfuse_dataset_name,
        preset_names=names,
    )
    merged = merge_preset_leaderboards(views)

    if include_diagnostics:
        diag = diagnostics_summary(df)
        if not diag.empty:
            if not merged.empty:
                merged = merged.merge(diag, on="llm", how="outer")
            else:
                merged = diag

    return merged
