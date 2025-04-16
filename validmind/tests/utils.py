# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Test Module Utils"""

import inspect
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from validmind.logging import get_logger

logger = get_logger(__name__)


def test_description(test_class: Type[Any], truncate: bool = True) -> str:
    description = inspect.getdoc(test_class).strip()

    if truncate and len(description.split("\n")) > 5:
        return description.strip().split("\n")[0] + "..."

    return description


def remove_nan_pairs(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    dataset_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove pairs where either true or predicted values are NaN/None.
    Args:
        y_true: List or array of true values
        y_pred: List or array of predicted values
        dataset_id: Optional identifier for the dataset (for logging)
    Returns:
        tuple: (cleaned_y_true, cleaned_y_pred)
    """
    # Convert to numpy arrays for easier handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Find indices where either value is NaN/None
    nan_mask = ~(pd.isnull(y_true) | pd.isnull(y_pred))
    nan_count = len(y_true) - np.sum(nan_mask)

    if nan_count > 0:
        dataset_info = f" from dataset '{dataset_id}'" if dataset_id else ""
        logger.warning(
            f"Found {nan_count} row(s){dataset_info} with NaN/None values. "
            f"Removing these pairs. {len(y_true)} -> {np.sum(nan_mask)} pairs remaining."
        )
        return y_true[nan_mask], y_pred[nan_mask]

    return y_true, y_pred


def ensure_equal_lengths(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    dataset_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check if true and predicted values have matching lengths, log warning if they don't,
    and truncate to the shorter length if necessary. Also removes any NaN/None values.

    Args:
        y_true: List or array of true values
        y_pred: List or array of predicted values
        dataset_id: Optional identifier for the dataset (for logging)

    Returns:
        tuple: (cleaned_y_true, cleaned_y_pred)
    """
    # First remove any NaN values
    y_true, y_pred = remove_nan_pairs(y_true, y_pred, dataset_id)

    # Then handle length mismatches
    if len(y_true) != len(y_pred):
        dataset_info = f" from dataset '{dataset_id}'" if dataset_id else ""
        min_length = min(len(y_true), len(y_pred))
        logger.warning(
            f"Length mismatch{dataset_info}: "
            f"true values ({len(y_true)}) != predicted values ({len(y_pred)}). "
            f"Truncating to first {min_length} pairs."
        )
        return y_true[:min_length], y_pred[:min_length]

    return y_true, y_pred


def validate_prediction(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    dataset_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Comprehensive validation of true and predicted value pairs.
    Handles NaN/None values and length mismatches.

    Args:
        y_true: List or array of true values
        y_pred: List or array of predicted values
        dataset_id: Optional identifier for the dataset (for logging)

    Returns:
        tuple: (cleaned_y_true, cleaned_y_pred) with matching lengths and no NaN values

    Example:
        >>> y_true, y_pred = validate_prediction_pairs(dataset.y, model.predict(dataset.X), dataset.input_id)
    """
    # First remove any NaN values
    y_true, y_pred = remove_nan_pairs(y_true, y_pred, dataset_id)

    # Then handle any length mismatches
    y_true, y_pred = ensure_equal_lengths(y_true, y_pred, dataset_id)

    return y_true, y_pred
