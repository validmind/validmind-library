from dataclasses import dataclass

import pandas as pd

from validmind.vm_models import Metric, ResultSummary, ResultTable, ResultTableMetadata


@dataclass
class TabularDescriptionTables(Metric):
    """
    Collects a set of descriptive statistics for a tabular dataset, for
    numerical, categorical and datetime variables.
    """

    name = "tabular_description_tables"
    required_context = ["dataset"]

    def description(self):
        return """
        This section provides descriptive statistics for numerical,
        categorical and datetime variables found in the dataset.
        """

    def get_summary_statistics_numerical(self, numerical_fields):
        summary_stats = self.df[numerical_fields].describe().T
        summary_stats["Missing Values (%)"] = (
            self.df[numerical_fields].isnull().mean() * 100
        )
        summary_stats["Data Type"] = self.df[numerical_fields].dtypes
        summary_stats = summary_stats[
            ["count", "mean", "min", "max", "Missing Values (%)", "Data Type"]
        ]
        summary_stats.columns = [
            "Num of Obs",
            "Mean",
            "Min",
            "Max",
            "Missing Values (%)",
            "Data Type",
        ]
        summary_stats["Num of Obs"] = summary_stats["Num of Obs"].astype(int)
        summary_stats = summary_stats.sort_values(
            by="Missing Values (%)", ascending=False
        )
        summary_stats.reset_index(inplace=True)
        summary_stats.rename(columns={"index": "Numerical Variable"}, inplace=True)
        return summary_stats

    def get_summary_statistics_categorical(self, categorical_fields):
        summary_stats = pd.DataFrame()
        if categorical_fields:  # check if the list is not empty
            for column in self.df[categorical_fields].columns:
                summary_stats.loc[column, "Num of Obs"] = int(self.df[column].count())
                summary_stats.loc[column, "Num of Unique Values"] = self.df[
                    column
                ].nunique()
                summary_stats.loc[column, "Unique Values"] = str(
                    self.df[column].unique()
                )
                summary_stats.loc[column, "Missing Values (%)"] = (
                    self.df[column].isnull().mean() * 100
                )
                summary_stats.loc[column, "Data Type"] = str(self.df[column].dtype)

            summary_stats = summary_stats.sort_values(
                by="Missing Values (%)", ascending=False
            )
            summary_stats.reset_index(inplace=True)
            summary_stats.rename(
                columns={"index": "Categorical Variable"}, inplace=True
            )
        return summary_stats

    def get_summary_statistics_datetime(self, datetime_fields):
        summary_stats = pd.DataFrame()
        for column in self.df[datetime_fields].columns:
            summary_stats.loc[column, "Num of Obs"] = int(self.df[column].count())
            summary_stats.loc[column, "Num of Unique Values"] = self.df[
                column
            ].nunique()
            summary_stats.loc[column, "Earliest Date"] = self.df[column].min()
            summary_stats.loc[column, "Latest Date"] = self.df[column].max()
            summary_stats.loc[column, "Missing Values (%)"] = (
                self.df[column].isnull().mean() * 100
            )
            summary_stats.loc[column, "Data Type"] = str(self.df[column].dtype)

        if not summary_stats.empty:
            summary_stats = summary_stats.sort_values(
                by="Missing Values (%)", ascending=False
            )
        summary_stats.reset_index(inplace=True)
        summary_stats.rename(columns={"index": "Datetime Variable"}, inplace=True)
        return summary_stats

    def summary(self, metric_value):
        summary_stats_numerical = metric_value["numerical"]
        summary_stats_categorical = metric_value["categorical"]
        summary_stats_datetime = metric_value["datetime"]

        return ResultSummary(
            results=[
                ResultTable(
                    data=summary_stats_numerical,
                    metadata=ResultTableMetadata(title="Numerical Variables"),
                ),
                ResultTable(
                    data=summary_stats_categorical,
                    metadata=ResultTableMetadata(title="Categorical Variables"),
                ),
                ResultTable(
                    data=summary_stats_datetime,
                    metadata=ResultTableMetadata(title="Datetime Variables"),
                ),
            ]
        )

    def get_categorical_columns(self):
        categorical_columns = self.df.select_dtypes(include="object").columns.tolist()
        return categorical_columns

    def get_numerical_columns(self):
        numerical_columns = self.df.select_dtypes(
            include=["int", "float"]
        ).columns.tolist()
        return numerical_columns

    def get_datetime_columns(self):
        datetime_columns = self.df.select_dtypes(include=["datetime"]).columns.tolist()
        return datetime_columns

    def run(self):
        numerical_fields = self.get_numerical_columns()
        categorical_fields = self.get_categorical_columns()
        datetime_fields = self.get_datetime_columns()

        summary_stats_numerical = self.get_summary_statistics_numerical(
            numerical_fields
        )
        summary_stats_categorical = self.get_summary_statistics_categorical(
            categorical_fields
        )
        summary_stats_datetime = self.get_summary_statistics_datetime(datetime_fields)

        return self.cache_results(
            {
                "numerical": summary_stats_numerical,
                "categorical": summary_stats_categorical,
                "datetime": summary_stats_datetime,
            }
        )