# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Time Series Test Suites
"""
from validmind.vm_models import TestSuite

from .statsmodels_timeseries import (
    RegressionModelDescription,
    RegressionModelsEvaluation,
)


class TimeSeriesDataQuality(TestSuite):
    """
    Test suite for data quality on time series datasets
    """

    suite_id = "time_series_data_quality"
    tests = [
        "validmind.data_validation.TimeSeriesOutliers",
        "validmind.data_validation.TimeSeriesMissingValues",
        "validmind.data_validation.TimeSeriesFrequency",
    ]


class TimeSeriesUnivariate(TestSuite):
    """
    This test suite provides a preliminary understanding of the target variable(s)
    used in the time series dataset. It visualizations that present the raw time
    series data and a histogram of the target variable(s).

    The raw time series data provides a visual inspection of the target variable's
    behavior over time. This helps to identify any patterns or trends in the data,
    as well as any potential outliers or anomalies. The histogram of the target
    variable displays the distribution of values, providing insight into the range
    and frequency of values observed in the data.
    """

    suite_id = "time_series_univariate"
    tests = [
        "validmind.data_validation.TimeSeriesLinePlot",
        "validmind.data_validation.TimeSeriesHistogram",
        "validmind.data_validation.ACFandPACFPlot",
        "validmind.data_validation.SeasonalDecompose",
        "validmind.data_validation.AutoSeasonality",
        "validmind.data_validation.AutoStationarity",
        "validmind.data_validation.RollingStatsPlot",
        "validmind.data_validation.AutoAR",
        "validmind.data_validation.AutoMA",
    ]


class TimeSeriesMultivariate(TestSuite):
    """
    This test suite provides a preliminary understanding of the features
    and relationship in multivariate dataset. It presents various
    multivariate visualizations that can help identify patterns, trends,
    and relationships between pairs of variables. The visualizations are
    designed to explore the relationships between multiple features
    simultaneously. They allow you to quickly identify any patterns or
    trends in the data, as well as any potential outliers or anomalies.
    The individual feature distribution can also be explored to provide
    insight into the range and frequency of values observed in the data.
    This multivariate analysis test suite aims to provide an overview of
    the data structure and guide further exploration and modeling.
    """

    suite_id = "time_series_multivariate"
    tests = [
        "validmind.data_validation.ScatterPlot",
        "validmind.data_validation.LaggedCorrelationHeatmap",
        "validmind.data_validation.EngleGrangerCoint",
        "validmind.data_validation.SpreadPlot",
    ]


class TimeSeriesDataset(TestSuite):
    """
    Test suite for time series datasets.
    """

    suite_id = "time_series_dataset"
    tests = [
        {
            "section_id": TimeSeriesDataQuality.suite_id,
            "section_description": TimeSeriesDataQuality.__doc__,
            "section_tests": TimeSeriesDataQuality.tests,
        },
        {
            "section_id": TimeSeriesUnivariate.suite_id,
            "section_description": TimeSeriesUnivariate.__doc__,
            "section_tests": TimeSeriesUnivariate.tests,
        },
        {
            "section_id": TimeSeriesMultivariate.suite_id,
            "section_description": TimeSeriesMultivariate.__doc__,
            "section_tests": TimeSeriesMultivariate.tests,
        },
    ]


class TimeSeriesModelValidation(TestSuite):
    """
    Test suite for time series model validation.
    """

    suite_id = "time_series_model_validation"
    tests = [
        {
            "section_id": RegressionModelDescription.suite_id,
            "section_description": RegressionModelDescription.__doc__,
            "section_tests": RegressionModelDescription.tests,
        },
        {
            "section_id": RegressionModelsEvaluation.suite_id,
            "section_description": RegressionModelsEvaluation.__doc__,
            "section_tests": RegressionModelsEvaluation.tests,
        },
    ]
