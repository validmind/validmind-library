# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Dataset class wrapper
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from validmind.logging import get_logger
from validmind.models import FunctionModel, PipelineModel
from validmind.vm_models.input import VMInput
from validmind.vm_models.model import VMModel

from .utils import ExtraColumns, as_df, compute_predictions, convert_index_to_datetime

logger = get_logger(__name__)


class VMDataset(VMInput):
    """Base class for VM datasets.

    Child classes should be used to support new dataset types (tensor, polars etc.)
    by converting the user's dataset into a numpy array collecting metadata like
    column names and then call this (parent) class `__init__` method.

    This way we can support multiple dataset types but under the hood we only
    need to work with numpy arrays and pandas dataframes in this class.

    Attributes:
        raw_dataset (np.ndarray): The raw dataset as a NumPy array.
        input_id (str): Identifier for the dataset.
        index (np.ndarray): The raw dataset index as a NumPy array.
        columns (Set[str]): The column names of the dataset.
        target_column (str): The target column name of the dataset.
        feature_columns (List[str]): The feature column names of the dataset.
        feature_columns_numeric (List[str]): The numeric feature column names of the dataset.
        feature_columns_categorical (List[str]): The categorical feature column names of the dataset.
        text_column (str): The text column name of the dataset for NLP tasks.
        target_class_labels (Dict): The class labels for the target columns.
        df (pd.DataFrame): The dataset as a pandas DataFrame.
        extra_columns (Dict): Extra columns to include in the dataset.
        copy_data (bool): Whether to copy the data. Defaults to True.
    """

    def __repr__(self):
        return f"VMDataset(input_id={self.input_id})"

    def __init__(
        self,
        raw_dataset: np.ndarray,
        input_id: str = None,
        model: VMModel = None,
        index: np.ndarray = None,
        index_name: str = None,
        date_time_index: bool = False,
        columns: list = None,
        target_column: str = None,
        feature_columns: list = None,
        text_column: str = None,
        extra_columns: dict = None,
        target_class_labels: dict = None,
        copy_data: bool = True,
    ):
        """
        Initializes a VMDataset instance.

        Args:
            raw_dataset (np.ndarray): The raw dataset as a NumPy array.
            input_id (str): Identifier for the dataset.
            model (VMModel): Model associated with the dataset.
            index (np.ndarray): The raw dataset index as a NumPy array.
            index_name (str): The raw dataset index name as a NumPy array.
            date_time_index (bool): Whether the index is a datetime index.
            columns (List[str], optional): The column names of the dataset. Defaults to None.
            target_column (str, optional): The target column name of the dataset. Defaults to None.
            feature_columns (str, optional): The feature column names of the dataset. Defaults to None.
            text_column (str, optional): The text column name of the dataset for nlp tasks. Defaults to None.
            target_class_labels (Dict, optional): The class labels for the target columns. Defaults to None.
            copy_data (bool, optional): Whether to copy the data. Defaults to True.
        """
        # initialize input_id
        self.input_id = input_id

        # initialize raw dataset
        if not isinstance(raw_dataset, np.ndarray):
            raise ValueError("Expected Numpy array for attribute raw_dataset")
        self._raw_dataset = raw_dataset

        # initialize index and index name
        if index is not None and not isinstance(index, np.ndarray):
            raise ValueError("Expected Numpy array for attribute raw_dataset")
        self.index = index

        self._df = pd.DataFrame(self._raw_dataset, columns=columns).infer_objects()
        # set index to dataframe
        if index is not None:
            self._df.set_index(pd.Index(index), inplace=True)
            self._df.index.name = index_name
        # attempt to convert index to datatime
        if date_time_index:
            self._df = convert_index_to_datetime(self._df)

        self.columns = columns or []
        self.column_aliases = {}
        self.target_column = target_column
        self.text_column = text_column
        self.target_class_labels = target_class_labels
        self.extra_columns = ExtraColumns.from_dict(extra_columns)
        self._set_feature_columns(feature_columns)
        self._copy_data = copy_data

        if model:
            self.assign_predictions(model)

    def _set_feature_columns(self, feature_columns=None):
        if feature_columns is not None and (
            not isinstance(feature_columns, list)
            or not all(isinstance(col, str) for col in feature_columns)
        ):
            raise ValueError("Expected list of column names for `feature_columns`")

        if feature_columns:
            self.feature_columns = feature_columns
        else:
            excluded = [self.target_column, *self.extra_columns.flatten()]
            self.feature_columns = [col for col in self.columns if col not in excluded]

        # Get dtypes without loading data into memory
        feature_dtypes = self._df[self.feature_columns].dtypes

        self.feature_columns_numeric = feature_dtypes[
            feature_dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))
        ].index.tolist()

        self.feature_columns_categorical = feature_dtypes[
            feature_dtypes.apply(
                lambda x: pd.api.types.is_categorical_dtype(x)
                or pd.api.types.is_object_dtype(x)
            )
        ].index.tolist()

    def _add_column(self, column_name, column_values):
        column_values = np.array(column_values)

        if column_values.ndim == 1:
            if len(column_values) != len(self._df):
                raise ValueError(
                    "Length of values doesn't match number of rows in the DataFrame."
                )
            self.columns.append(column_name)
            self._df[column_name] = column_values
        elif column_values.ndim == 2:
            if column_values.shape[0] != len(self._df):
                raise ValueError(
                    "Number of rows in values doesn't match number of rows in the DataFrame."
                )
            self.columns.append(column_name)
            self._df[column_name] = column_values.tolist()

        else:
            raise ValueError("Only 1D and 2D arrays are supported for column_values.")

    def _validate_assign_predictions(
        self,
        model: VMModel,
        prediction_column: str,
        prediction_values: list,
        probability_column: str,
        probability_values: list,
    ):
        if not isinstance(model, VMModel):
            raise ValueError("Expected VMModel instance for argument `model`")

        if prediction_column and prediction_values is not None:
            raise ValueError(
                "Only one of the following arguments can be provided: "
                "`prediction_column`, `prediction_values`"
            )

        if probability_column and probability_values is not None:
            raise ValueError(
                "Only one of the following arguments can be provided: "
                "`probability_column`, `probability_values`"
            )

        if prediction_column and prediction_column not in self.columns:
            raise ValueError(
                f"Prediction column {prediction_column} doesn't exist in the dataset"
            )

        if probability_column and probability_column not in self.columns:
            raise ValueError(
                f"Probability column {probability_column} doesn't exist in the dataset"
            )

        if (probability_column or probability_values is not None) and (
            not prediction_column and prediction_values is None
        ):
            raise ValueError(
                "Cannot use precomputed probabilities without precomputed predictions"
            )

    def with_options(self, **kwargs: Dict[str, Any]) -> "VMDataset":
        """Support options provided when passing an input to run_test or run_test_suite

        Example:
        ```python
        # to only use a certain subset of columns in the dataset:
        run_test(
            "validmind.SomeTestID",
            inputs={
                "dataset": {
                    "input_id": "my_dataset_id",
                    "columns": ["col1", "col2"],
                }
            }
        )

        # behind the scenes, this retrieves the dataset object (VMDataset) from the registry
        # and then calls the `with_options()` method and passes `{"columns": ...}`
        ```

        Args:
            **kwargs: Options:
                - columns: Filter columns in the dataset

        Returns:
            VMDataset: A new instance of the dataset with only the specified columns
        """
        if "columns" in kwargs:
            # filter columns (create a temp copy of self with only specified columns)
            # TODO: need a more robust mechanism for this as we expand on this feature
            columns = kwargs.pop("columns")

            new = deepcopy(self)

            new._set_feature_columns(
                [col for col in new.feature_columns if col in columns]
            )
            new.text_column = new.text_column if new.text_column in columns else None
            new.target_column = (
                new.target_column if new.target_column in columns else None
            )
            new.extra_columns.extras = new.extra_columns.extras.intersection(columns)

            return new

        if kwargs:
            raise NotImplementedError(
                f"Options {kwargs} are not supported for this input"
            )

    def _handle_deprecated_parameters(
        self, prediction_probabilities, probability_values
    ):
        """Handle deprecated parameters and return the correct probability values."""
        if prediction_probabilities is not None:
            warnings.warn(
                "The `prediction_probabilities` argument is deprecated. Use `probability_values` instead.",
                DeprecationWarning,
            )
            return prediction_probabilities
        return probability_values

    def _check_existing_predictions(self, model):
        """Check for existing predictions and probabilities, warn if overwriting."""
        if self.prediction_column(model):
            logger.warning("Model predictions already assigned... Overwriting.")

        if self.probability_column(model):
            logger.warning("Model probabilities already assigned... Overwriting.")

    def _get_precomputed_values(self, prediction_column, probability_column):
        """Get precomputed prediction and probability values from existing columns."""
        prediction_values = None
        probability_values = None

        if prediction_column:
            prediction_values = self._df[prediction_column].values

            if probability_column:
                probability_values = self._df[probability_column].values

        return prediction_values, probability_values

    def _compute_predictions_if_needed(self, model, prediction_values, **kwargs):
        """Compute predictions if not provided."""
        if prediction_values is None:
            X = self.df if isinstance(model, (FunctionModel, PipelineModel)) else self.x
            return compute_predictions(model, X, **kwargs)
        return None, prediction_values

    def _handle_dictionary_predictions(self, model, prediction_values):
        """Handle dictionary predictions by converting to separate columns."""
        if (
            prediction_values is not None
            and len(prediction_values) > 0
            and isinstance(prediction_values[0], dict)
        ):
            df_prediction_values = pd.DataFrame.from_dict(
                prediction_values, orient="columns"
            )

            for column_name in df_prediction_values.columns.tolist():
                values = df_prediction_values[column_name].values

                if column_name == "prediction":
                    prediction_column = f"{model.input_id}_prediction"
                    self._add_column(prediction_column, values)
                    self.prediction_column(model, prediction_column)
                else:
                    self._add_column(f"{model.input_id}_{column_name}", values)

            return (
                True,
                None,
            )  # Return True to indicate dictionary handled, None for prediction_column
        return False, None

    def _add_prediction_columns(
        self,
        model,
        prediction_column,
        prediction_values,
        probability_column,
        probability_values,
    ):
        """Add prediction and probability columns to the dataset."""
        if prediction_column is None:
            prediction_column = f"{model.input_id}_prediction"

        self._add_column(prediction_column, prediction_values)
        self.prediction_column(model, prediction_column)

        if probability_values is not None:
            if probability_column is None:
                probability_column = f"{model.input_id}_probabilities"
            self._add_column(probability_column, probability_values)
            self.probability_column(model, probability_column)
        else:
            logger.info(
                "No probabilities computed or provided. "
                "Not adding probability column to the dataset."
            )

    def assign_predictions(
        self,
        model: VMModel,
        prediction_column: Optional[str] = None,
        prediction_values: Optional[Any] = None,
        probability_column: Optional[str] = None,
        probability_values: Optional[Any] = None,
        prediction_probabilities: Optional[
            Any
        ] = None,  # DEPRECATED: use probability_values
        **kwargs: Dict[str, Any],
    ) -> None:
        """Assign predictions and probabilities to the dataset.

        Args:
            model (VMModel): The model used to generate the predictions.
            prediction_column (Optional[str]): The name of the column containing the predictions.
            prediction_values (Optional[Any]): The values of the predictions. Can be array-like (list, numpy array, pandas Series, etc.).
            probability_column (Optional[str]): The name of the column containing the probabilities.
            probability_values (Optional[Any]): The values of the probabilities. Can be array-like (list, numpy array, pandas Series, etc.).
            prediction_probabilities (Optional[Any]): DEPRECATED: The values of the probabilities. Use probability_values instead.
            **kwargs: Additional keyword arguments that will get passed through to the model's `predict` method.
        """
        # Handle deprecated parameters
        probability_values = self._handle_deprecated_parameters(
            prediction_probabilities, probability_values
        )

        # Convert pandas Series to numpy array for prediction_values
        if (
            hasattr(prediction_values, "values")
            and hasattr(prediction_values, "index")
            and hasattr(prediction_values, "dtype")
        ):
            prediction_values = prediction_values.values

        # Convert pandas Series to numpy array for probability_values
        if (
            hasattr(probability_values, "values")
            and hasattr(probability_values, "index")
            and hasattr(probability_values, "dtype")
        ):
            probability_values = probability_values.values

        # Validate input parameters
        self._validate_assign_predictions(
            model,
            prediction_column,
            prediction_values,
            probability_column,
            probability_values,
        )

        # Check for existing predictions and warn if overwriting
        self._check_existing_predictions(model)

        # Get precomputed values if column names are provided
        if prediction_column or probability_column:
            prediction_values, prob_values_from_column = self._get_precomputed_values(
                prediction_column, probability_column
            )
            if prob_values_from_column is not None:
                probability_values = prob_values_from_column

        # Compute predictions if not provided
        if prediction_values is None:
            probability_values, prediction_values = self._compute_predictions_if_needed(
                model, prediction_values, **kwargs
            )

        # Handle dictionary predictions
        is_dict_handled, _ = self._handle_dictionary_predictions(
            model, prediction_values
        )

        # Add prediction and probability columns (skip if dictionary was handled)
        if not is_dict_handled:
            self._add_prediction_columns(
                model,
                prediction_column,
                prediction_values,
                probability_column,
                probability_values,
            )

    def prediction_column(self, model: VMModel, column_name: str = None) -> str:
        """Get or set the prediction column for a model."""
        if column_name and column_name not in self.columns:
            raise ValueError("{column_name} doesn't exist in the dataset")

        if column_name and column_name in self.feature_columns:
            self.feature_columns.remove(column_name)
            self._set_feature_columns(self.feature_columns)

        return self.extra_columns.prediction_column(model, column_name)

    def probability_column(self, model: VMModel, column_name: str = None) -> str:
        """Get or set the probability column for a model."""
        if column_name and column_name not in self.columns:
            raise ValueError("{column_name} doesn't exist in the dataset")

        if column_name and column_name in self.feature_columns:
            self.feature_columns.remove(column_name)
            self._set_feature_columns(self.feature_columns)

        return self.extra_columns.probability_column(model, column_name)

    def assign_scores(
        self,
        metrics: Union[str, List[str]],
        model: Optional[VMModel] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Assign computed row metric scores to the dataset as new columns.

        This method computes row metrics for the given model and dataset, then adds
        the computed scores as new columns to the dataset using the naming convention:
        {model.input_id}_{metric_name}

        Args:
            model (Optional[VMModel]): Optional model used to compute the scores. If provided and
                it has a valid `input_id`, that will be used as a prefix for column names.
                If not provided (or no `input_id`), columns will be created without a prefix.
            metrics (Union[str, List[str]]): Single metric ID or list of metric IDs.
                Can be either:
                - Short name (e.g., "BrierScore", "LogLoss")
                - Full metric ID (e.g., "validmind.scorer.classification.BrierScore")
            **kwargs: Additional parameters passed to the row metrics.

        Examples:
            # Single metric
            dataset.assign_scores(model, "BrierScore")

            # Multiple metrics
            dataset.assign_scores(model, ["BrierScore", "LogLoss"])

            # With parameters
            dataset.assign_scores(model, "ClassBalance", threshold=0.5)

        Raises:
            ValueError: If metric computation fails.
            ImportError: If scorer module cannot be imported.
        """
        model_input_id = None
        if model is not None:
            model_input_id = getattr(model, "input_id", None)
            if not model_input_id:
                logger.warning(
                    "Model has no input_id; creating score columns without prefix."
                )

        # Normalize metrics to a list
        if isinstance(metrics, str):
            metrics = [metrics]

        # Process each metric
        for metric in metrics:
            self._assign_single_score(metric, model, model_input_id, kwargs)

    def _assign_single_score(
        self,
        metric: str,
        model: Optional[VMModel],
        model_input_id: Optional[str],
        params: Dict[str, Any],
    ) -> None:
        """Compute and add a single metric's scores as dataset columns."""
        # Import scorer module
        try:
            from validmind.scorer import run_scorer
        except ImportError as e:
            raise ImportError(
                f"Failed to import scorer module: {e}. "
                "Make sure validmind.scorer is available."
            ) from e

        # Normalize metric ID and name
        metric_id = self._normalize_metric_id(metric)
        metric_name = self._extract_metric_name(metric_id)
        column_name = self._build_score_column_name(model_input_id, metric_name)

        try:
            inputs = {"dataset": self}
            if model is not None:
                inputs["model"] = model
            result = run_scorer(
                metric_id,
                inputs=inputs,
                params=params,
                show=False,
            )

            if result.raw_data and hasattr(result.raw_data, "scorer_output"):
                scorer_output = result.raw_data.scorer_output
                self._process_and_add_scorer_output(
                    scorer_output, model_input_id, metric_name
                )
            else:
                column_values = self._process_metric_value(result.metric)
                self.add_extra_column(column_name, column_values)

            logger.info(f"Added metric column(s) for '{metric_name}'")
        except Exception as e:
            logger.error(f"Failed to compute metric {metric_id}: {e}")
            raise ValueError(f"Failed to compute metric {metric_id}: {e}") from e

    def _process_and_add_scorer_output(
        self, scorer_output: Any, model_input_id: Optional[str], metric_name: str
    ) -> None:
        """Process scorer output and add appropriate columns to the dataset.

        Args:
            scorer_output: The raw scorer output (list, scalar, list of dicts, etc.)
            model_input_id: The model input ID for column naming
            metric_name: The metric name for column naming

        Raises:
            ValueError: If scorer output length doesn't match dataset length or
                       if list of dictionaries has inconsistent keys
        """
        if isinstance(scorer_output, list):
            self._process_list_scorer_output(scorer_output, model_input_id, metric_name)
        elif np.isscalar(scorer_output):
            self._process_scalar_scorer_output(
                scorer_output, model_input_id, metric_name
            )
        else:
            self._process_other_scorer_output(
                scorer_output, model_input_id, metric_name
            )

    def _process_list_scorer_output(
        self, scorer_output: list, model_input_id: Optional[str], metric_name: str
    ) -> None:
        """Process list scorer output and add appropriate columns."""
        if len(scorer_output) != len(self._df):
            raise ValueError(
                f"Scorer output length {len(scorer_output)} does not match dataset length {len(self._df)}"
            )

        if scorer_output and isinstance(scorer_output[0], dict):
            self._process_dict_list_scorer_output(
                scorer_output, model_input_id, metric_name
            )
        else:
            self._process_regular_list_scorer_output(
                scorer_output, model_input_id, metric_name
            )

    def _process_dict_list_scorer_output(
        self, scorer_output: list, model_input_id: Optional[str], metric_name: str
    ) -> None:
        """Process list of dictionaries scorer output."""
        # Validate that all dictionaries have the same keys
        first_keys = set(scorer_output[0].keys())
        for i, item in enumerate(scorer_output):
            if not isinstance(item, dict):
                raise ValueError(
                    f"All items in list must be dictionaries, but item at index {i} is {type(item)}"
                )
            if set(item.keys()) != first_keys:
                raise ValueError(
                    f"All dictionaries must have the same keys. "
                    f"First dict has keys {sorted(first_keys)}, "
                    f"but dict at index {i} has keys {sorted(item.keys())}"
                )

        # Add a column for each key in the dictionaries
        for key in first_keys:
            column_name = self._build_score_column_name(
                model_input_id, metric_name, key
            )
            column_values = np.array([item[key] for item in scorer_output])
            self.add_extra_column(column_name, column_values)
            logger.info(f"Added metric column '{column_name}'")

    def _process_regular_list_scorer_output(
        self, scorer_output: list, model_input_id: Optional[str], metric_name: str
    ) -> None:
        """Process regular list scorer output."""
        column_name = self._build_score_column_name(model_input_id, metric_name)
        column_values = np.array(scorer_output)
        self.add_extra_column(column_name, column_values)
        logger.info(f"Added metric column '{column_name}'")

    def _process_scalar_scorer_output(
        self, scorer_output: Any, model_input_id: Optional[str], metric_name: str
    ) -> None:
        """Process scalar scorer output."""
        column_name = self._build_score_column_name(model_input_id, metric_name)
        column_values = np.full(len(self._df), scorer_output)
        self.add_extra_column(column_name, column_values)
        logger.info(f"Added metric column '{column_name}'")

    def _process_other_scorer_output(
        self, scorer_output: Any, model_input_id: Optional[str], metric_name: str
    ) -> None:
        """Process other types of scorer output."""
        try:
            output_array = np.array(scorer_output)
            if len(output_array) != len(self._df):
                raise ValueError(
                    f"Scorer output length {len(output_array)} does not match dataset length {len(self._df)}"
                )
            column_name = self._build_score_column_name(model_input_id, metric_name)
            self.add_extra_column(column_name, output_array)
            logger.info(f"Added metric column '{column_name}'")
        except Exception as e:
            raise ValueError(f"Could not process scorer output: {e}") from e

    def _build_score_column_name(
        self, model_input_id: Optional[str], metric_name: str, key: Optional[str] = None
    ) -> str:
        """Build a score column name with optional model prefix and optional key suffix.

        Args:
            model_input_id: Optional model input_id to prefix the column name.
            metric_name: The metric name.
            key: Optional sub-key to append (for dict outputs).

        Returns:
            str: The constructed column name.
        """
        parts: List[str] = []
        if model_input_id:
            parts.append(model_input_id)
        parts.append(metric_name)
        if key:
            parts.append(str(key))
        return "_".join(parts)

    def _process_scorer_output(self, scorer_output: Any) -> np.ndarray:
        """Process scorer output and return column values for the dataset.

        Args:
            scorer_output: The raw scorer output (list, scalar, etc.)

        Returns:
            np.ndarray: Column values for the dataset

        Raises:
            ValueError: If scorer output length doesn't match dataset length
        """
        if isinstance(scorer_output, list):
            # List output - should be one value per row
            if len(scorer_output) != len(self._df):
                raise ValueError(
                    f"Scorer output length {len(scorer_output)} does not match dataset length {len(self._df)}"
                )
            return np.array(scorer_output)
        elif np.isscalar(scorer_output):
            # Scalar output - repeat for all rows
            return np.full(len(self._df), scorer_output)
        else:
            # Other types - try to convert to array
            try:
                output_array = np.array(scorer_output)
                if len(output_array) != len(self._df):
                    raise ValueError(
                        f"Scorer output length {len(output_array)} does not match dataset length {len(self._df)}"
                    )
                return output_array
            except Exception as e:
                raise ValueError(f"Could not process scorer output: {e}") from e

    def _process_metric_value(self, metric_value: Any) -> np.ndarray:
        """Process metric value and return column values for the dataset.

        Args:
            metric_value: The metric value to process (could be MetricValues object or raw value)

        Returns:
            np.ndarray: Column values for the dataset

        Raises:
            ValueError: If metric value length doesn't match dataset length
        """
        # Handle None case (some tests don't return metric values)
        if metric_value is None:
            # Return zeros for all rows as a default
            return np.zeros(len(self._df))

        # Handle different metric value types
        if hasattr(metric_value, "get_values"):
            # New MetricValues object (UnitMetricValue or RowMetricValues)
            values = metric_value.get_values()
            if metric_value.is_list():
                # Row metrics - should be one value per row
                if len(values) != len(self._df):
                    raise ValueError(
                        f"Row metric value length {len(values)} does not match dataset length {len(self._df)}"
                    )
                return np.array(values)
            else:
                # Unit metrics - repeat scalar value for all rows
                return np.full(len(self._df), values)
        elif np.isscalar(metric_value):
            # Legacy scalar value - repeat for all rows
            return np.full(len(self._df), metric_value)
        else:
            # Legacy list value - use directly
            if len(metric_value) != len(self._df):
                raise ValueError(
                    f"Metric value length {len(metric_value)} does not match dataset length {len(self._df)}"
                )
            return np.array(metric_value)

    def _normalize_metric_id(self, metric: str) -> str:
        """Normalize metric identifier to full validmind row metric ID.

        Args:
            metric (str): Metric identifier (short name or full ID)

        Returns:
            str: Full metric ID
        """
        # If already a full ID, return as-is
        if metric.startswith("validmind.scorer."):
            return metric

        # Try to find the metric by short name
        try:
            from validmind.scorer import list_scorers
            from validmind.tests._store import scorer_store

            # Get built-in scorers
            available_metrics = list_scorers()

            # Add custom scorers from scorer store
            # Register custom metric if not already in scorer store
            if metric not in scorer_store.scorers:
                scorer_store.register_scorer(metric)
            all_scorers = list(scorer_store.scorers.keys())
            # Find metrics in custom_scorers that aren't already in available_metrics
            new_metrics = [m for m in all_scorers if m not in available_metrics]
            available_metrics.extend(new_metrics)

            # Look for exact match with short name
            for metric_id in available_metrics:
                if metric_id == metric:
                    return metric_id

            # If no exact match found, raise error with suggestions
            suggestions = [m for m in available_metrics if metric.lower() in m.lower()]
            if suggestions:
                raise ValueError(
                    f"Metric '{metric}' not found in scorer. Did you mean one of: {suggestions[:5]}"
                )
            else:
                raise ValueError(
                    f"Metric '{metric}' not found in scorer. Available metrics: {available_metrics[:10]}..."
                )

        except ImportError as e:
            raise ImportError(f"Failed to import scorer for metric lookup: {e}") from e

    def _extract_metric_name(self, metric_id: str) -> str:
        """Extract the metric name from a full metric ID.

        Args:
            metric_id (str): Full metric ID

        Returns:
            str: Metric name
        """
        # Extract the last part after the final dot
        return metric_id.split(".")[-1]

    def add_extra_column(self, column_name, column_values=None):
        """Adds an extra column to the dataset without modifying the dataset `features` and `target` columns.

        Args:
            column_name (str): The name of the extra column.
            column_values (np.ndarray, optional): The values of the extra column.
        """
        if column_name not in self.columns and (
            column_values is None or len(column_values) == 0
        ):
            raise ValueError(
                "Column values must be provided when the column doesn't exist in the dataset"
            )

        # some warnings to let the user know what's happening
        if column_name in self.extra_columns:
            logger.warning(f"{column_name} is already an extra column. Overwriting...")
        elif column_name in self.columns and column_values:
            logger.warning(
                f"{column_name} already exists in the dataset but `column_values` were passed. Overwriting..."
            )

        self.extra_columns.extras.add(column_name)
        self._add_column(column_name, column_values)

        # reset feature columns to exclude the new extra column
        self._set_feature_columns()

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns the dataset as a pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        # only include feature, text and target columns
        # don't include internal pred and prob columns
        columns = self.feature_columns.copy()

        # text column can also be a feature column so don't add it twice
        if self.text_column and self.text_column not in columns:
            columns.append(self.text_column)

        if self.extra_columns.extras:
            # add user-defined extra columns
            columns.extend(self.extra_columns.extras)

        if self.target_column:
            # shouldn't be a feature column but add this to be safe
            assert self.target_column not in columns
            columns.append(self.target_column)

        # Check if all columns in self._df are requested
        all_columns = set(columns) == set(self._df.columns)

        # For copy_data=False and all columns: return exact same DataFrame object
        if not self._copy_data and all_columns:
            return self._df
        # For copy_data=False and subset of columns: return view with shared data
        elif not self._copy_data:
            return as_df(self._df[columns])
        # For copy_data=True: return independent copy with duplicated data
        else:
            return as_df(self._df[columns]).copy()

    @property
    def x(self) -> np.ndarray:
        """
        Returns the input features (X) of the dataset.

        Returns:
            np.ndarray: The input features.
        """
        return self._df[self.feature_columns].to_numpy()

    @property
    def y(self) -> np.ndarray:
        """
        Returns the target variables (y) of the dataset.

        Returns:
            np.ndarray: The target variables.
        """
        return self._df[self.target_column].to_numpy()

    def y_pred(self, model) -> np.ndarray:
        """Returns the predictions for a given model.

        Attempts to stack complex prediction types (e.g., embeddings) into a single,
        multi-dimensional array.

        Args:
            model (VMModel): The model whose predictions are sought.

        Returns:
            np.ndarray: The predictions for the model
        """
        pred_col = self.prediction_column(model)

        if pred_col is None:
            raise ValueError(f"No prediction column found for model `{model.input_id}`")

        return np.stack(self._df[pred_col].values)

    def y_prob(self, model) -> np.ndarray:
        """Returns the probabilities for a given model.

        Args:
            model (str): The ID of the model whose predictions are sought.

        Returns:
            np.ndarray: The probability variables.
        """
        prob_col = self.probability_column(model)

        if prob_col is None:
            raise ValueError(
                f"No probability column found for model `{model.input_id}`"
            )

        return self._df[prob_col].values

    def x_df(self):
        """Returns a dataframe containing only the feature columns"""
        return as_df(self._df[self.feature_columns])

    def y_df(self) -> pd.DataFrame:
        """Returns a dataframe containing the target column"""
        return as_df(self._df[self.target_column])

    def y_pred_df(self, model) -> pd.DataFrame:
        """Returns a dataframe containing the predictions for a given model"""
        pred_col = self.prediction_column(model)

        if pred_col is None:
            raise ValueError(f"No prediction column found for model `{model.input_id}`")

        return as_df(self._df[pred_col])

    def y_prob_df(self, model) -> pd.DataFrame:
        """Returns a dataframe containing the probabilities for a given model"""
        prob_col = self.probability_column(model)

        if prob_col is None:
            raise ValueError(
                f"No probability column found for model `{model.input_id}`"
            )

        return as_df(self._df[prob_col])

    def target_classes(self):
        """Returns the target class labels or unique values of the target column."""
        return self.target_class_labels or [str(i) for i in np.unique(self.y)]

    def __str__(self):
        return (
            f"=================\n"
            f"VMDataset object: \n"
            f"=================\n"
            f"Input ID: {self.input_id}\n"
            f"Target Column: {self.target_column}\n"
            f"Feature Columns: {self.feature_columns}\n"
            f"Text Column: {self.text_column}\n"
            f"Extra Columns: {self.extra_columns}\n"
            f"Target Class Labels: {self.target_class_labels}\n"
            f"Columns: {self.columns}\n"
            f"Index: {self.index}\n"
            f"=================\n"
        )


class DataFrameDataset(VMDataset):
    """
    VM dataset implementation for pandas DataFrame.
    """

    def __init__(
        self,
        raw_dataset: pd.DataFrame,
        input_id: str = None,
        model: VMModel = None,
        target_column: str = None,
        extra_columns: dict = None,
        feature_columns: list = None,
        text_column: str = None,
        target_class_labels: dict = None,
        date_time_index: bool = False,
        copy_data: bool = True,
    ):
        """
        Initializes a DataFrameDataset instance, preserving original pandas dtypes.

        Args:
            raw_dataset (pd.DataFrame): The raw dataset as a pandas DataFrame.
            input_id (str, optional): Identifier for the dataset. Defaults to None.
            model (VMModel, optional): Model associated with the dataset. Defaults to None.
            target_column (str, optional): The target column of the dataset. Defaults to None.
            extra_columns (dict, optional): Extra columns to include in the dataset. Defaults to None.
            feature_columns (list, optional): The feature columns of the dataset. Defaults to None.
            text_column (str, optional): The text column name of the dataset for NLP tasks. Defaults to None.
            target_class_labels (dict, optional): The class labels for the target columns. Defaults to None.
            date_time_index (bool, optional): Whether to use date-time index. Defaults to False.
            copy_data (bool, optional): Whether to create a copy of the input data. Defaults to True.
        """

        VMInput.__init__(self)

        self.input_id = input_id

        index = None
        if isinstance(raw_dataset.index, pd.Index):
            index = raw_dataset.index.values
        self.index = index

        # Store the DataFrame directly
        self._df = raw_dataset

        if date_time_index:
            self._df = convert_index_to_datetime(self._df)

        self.columns = raw_dataset.columns.tolist()
        self.column_aliases = {}
        self.target_column = target_column
        self.text_column = text_column
        self.target_class_labels = target_class_labels
        self.extra_columns = ExtraColumns.from_dict(extra_columns)
        self._copy_data = copy_data

        # Add warning when copy_data is False
        if not copy_data:
            logger.warning(
                "Dataset initialized with copy_data=False. Changes to the original DataFrame "
                "may affect this dataset. Use this option only when memory efficiency is critical "
                "and you won't modify the source data."
            )

        self._set_feature_columns(feature_columns)

        if model:
            self.assign_predictions(model)


class PolarsDataset(VMDataset):
    """
    VM dataset implementation for Polars DataFrame.
    """

    def __init__(
        self,
        raw_dataset: pl.DataFrame,
        input_id: str = None,
        model: VMModel = None,
        target_column: str = None,
        extra_columns: dict = None,
        feature_columns: list = None,
        text_column: str = None,
        target_class_labels: dict = None,
        date_time_index: bool = False,
    ):
        """
        Initializes a PolarsDataset instance.

        Args:
            raw_dataset (pl.DataFrame): The raw dataset as a Polars DataFrame.
            input_id (str, optional): Identifier for the dataset. Defaults to None.
            model (VMModel, optional): Model associated with the dataset. Defaults to None.
            target_column (str, optional): The target column of the dataset. Defaults to None.
            extra_columns (dict, optional): Extra columns to include in the dataset. Defaults to None.
            feature_columns (list, optional): The feature columns of the dataset. Defaults to None.
            text_column (str, optional): The text column name of the dataset for NLP tasks. Defaults to None.
            target_class_labels (dict, optional): The class labels for the target columns. Defaults to None.
            date_time_index (bool, optional): Whether to use date-time index. Defaults to False.
        """
        super().__init__(
            raw_dataset=raw_dataset.to_numpy(),
            input_id=input_id,
            model=model,
            index_name=None,
            index=None,
            columns=raw_dataset.columns,
            target_column=target_column,
            extra_columns=extra_columns,
            feature_columns=feature_columns,
            text_column=text_column,
            target_class_labels=target_class_labels,
            date_time_index=date_time_index,
        )


class TorchDataset(VMDataset):
    """
    VM dataset implementation for PyTorch Datasets.
    """

    def __init__(
        self,
        raw_dataset,
        input_id: str = None,
        model: VMModel = None,
        index_name=None,
        index=None,
        columns=None,
        target_column: str = None,
        extra_columns: dict = None,
        feature_columns: list = None,
        text_column: str = None,
        target_class_labels: dict = None,
    ):
        """
        Initializes a TorchDataset instance.

        Args:
            raw_dataset (torch.utils.data.Dataset): The raw dataset as a PyTorch Dataset.
            index_name (str): The raw dataset index name.
            index (np.ndarray): The raw dataset index as a NumPy array.
            columns (List[str]): The column names of the dataset.
            target_column (str, optional): The target column of the dataset. Defaults to None.
            feature_columns (list, optional): The feature columns of the dataset. Defaults to None.
            text_column (str, optional): The text column name of the dataset for nlp tasks. Defaults to None.
            target_class_labels (Dict, optional): The class labels for the target columns. Defaults to None.
        """

        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is not installed, please run `pip install validmind[pytorch]`"
            )

        columns = []

        for id, tens in zip(range(0, len(raw_dataset.tensors)), raw_dataset.tensors):
            if id == 0 and feature_columns is None:
                n_cols = tens.shape[1]
                feature_columns = [
                    "x" + feature_id
                    for feature_id in np.linspace(
                        0, n_cols - 1, num=n_cols, dtype=int
                    ).astype(str)
                ]
                columns.append(feature_columns)

            elif id == 1 and target_column is None:
                target_column = "y"
                columns.append(target_column)

            elif id == 2 and extra_columns is None:
                extra_columns.prediction_column = "y_pred"
                columns.append(extra_columns.prediction_column)

        merged_tensors = torch.cat(raw_dataset.tensors, dim=1).numpy()

        super().__init__(
            input_id=input_id,
            raw_dataset=merged_tensors,
            model=model,
            index_name=index_name,
            index=index,
            columns=columns,
            target_column=target_column,
            feature_columns=feature_columns,
            text_column=text_column,
            extra_columns=extra_columns,
            target_class_labels=target_class_labels,
        )
