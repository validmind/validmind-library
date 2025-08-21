# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
PII filtering utilities using Microsoft Presidio for detecting and masking
personally identifiable information in test result data.
"""

import os
from enum import Enum
from typing import Dict

import pandas as pd

from ...logging import get_logger

logger = get_logger(__name__)


class PIIDetectionMode(Enum):
    """Enum for PII detection modes."""

    DISABLED = "disabled"
    TEST_RESULTS = "test_results"
    TEST_DESCRIPTIONS = "test_descriptions"
    ALL = "all"


# Default entities to detect common PII types
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "US_PASSPORT",
    "US_BANK_NUMBER",
    "IBAN_CODE",
]

# Default confidence threshold
DEFAULT_THRESHOLD = 0.5

# Default sample size for DataFrame PII scanning
SAMPLE_SIZE = 100


def get_pii_detection_mode() -> PIIDetectionMode:
    """
    Get the current PII detection mode.

    Returns:
        PIIDetectionMode.DISABLED if:
        - Environment variable is not set
        - Environment variable is set to "disabled"
        - Presidio packages are not installed
        - Invalid mode value

        Otherwise returns the specified mode (test_results, test_descriptions, or all)
    """
    mode_str = os.getenv("VALIDMIND_PII_DETECTION", "disabled").lower()

    try:
        mode = PIIDetectionMode(mode_str)
    except ValueError:
        logger.warning(
            f"Invalid PII detection mode '{mode_str}'. "
            f"Valid options: {', '.join([mode.value for mode in PIIDetectionMode])}. "
            f"Defaulting to 'disabled'."
        )
        mode = PIIDetectionMode.DISABLED

    # If mode is not disabled, check if Presidio is actually available
    if mode != PIIDetectionMode.DISABLED:
        if not _is_presidio_available():
            logger.warning(
                f"PII detection mode '{mode.value}' requested but Presidio not available. "
                "Falling back to 'disabled' mode. Install with: pip install validmind[pii-detection]"
            )
            mode = PIIDetectionMode.DISABLED

    return mode


def _is_presidio_available() -> bool:
    """Check if any Presidio components are available."""
    return _get_presidio_text() is not None or _get_presidio_df() is not None


def _get_presidio_text():
    """Get Presidio analyzer for text analysis."""
    from presidio_analyzer import AnalyzerEngine

    return AnalyzerEngine()


def _get_presidio_df():
    """Get Presidio Structured PandasAnalysisBuilder for DataFrame analysis."""
    from presidio_structured import PandasAnalysisBuilder

    return PandasAnalysisBuilder()


def scan_text(text: str) -> bool:
    """
    Scan text for PII content. Raises ValueError if PII is found.

    Args:
        text: The text to scan for PII

    Returns:
        True if no PII is found

    Raises:
        ValueError: If PII is detected
    """
    # sanity check
    mode = get_pii_detection_mode()
    if mode == PIIDetectionMode.DISABLED:
        return True

    analyzer = _get_presidio_text()
    results = analyzer.analyze(text=text, entities=DEFAULT_ENTITIES, language="en")

    # Filter results by confidence threshold
    pii_entities = [
        {
            "entity_type": result.entity_type,
            "start": result.start,
            "end": result.end,
            "score": result.score,
            "text": text[result.start : result.end],
        }
        for result in results
        if result.score >= DEFAULT_THRESHOLD
    ]

    if pii_entities:
        entity_types = set(entity["entity_type"] for entity in pii_entities)
        raise ValueError(
            f"PII detected in text content. Entity types found: {', '.join(entity_types)}."
        )

    return True


def scan_df(df: pd.DataFrame) -> bool:
    """
    Scan a pandas DataFrame for PII content. Raises ValueError if PII is found.

    Args:
        df: The DataFrame to scan

    Returns:
        True if no PII is found

    Raises:
        ValueError: If PII is detected
    """
    # sanity check
    mode = get_pii_detection_mode()
    if mode == PIIDetectionMode.DISABLED:
        return True

    # Scan all string/object columns
    columns = [col for col in df.columns if df[col].dtype == "object"]

    if not columns:
        return True

    # Limit the number of rows to scan for performance
    sample_df = df.head(SAMPLE_SIZE) if len(df) > SAMPLE_SIZE else df

    # Use structured analysis
    builder = _get_presidio_df()
    tabular_analysis = builder.generate_analysis(
        sample_df,
        selection_strategy="mixed",
        mixed_strategy_threshold=DEFAULT_THRESHOLD,
    )

    entity_mapping: Dict[str, str] = getattr(tabular_analysis, "entity_mapping", {})

    pii_columns = [
        column
        for column in columns
        if column in entity_mapping and entity_mapping[column]
    ]

    if pii_columns:
        entity_types = [entity_mapping[col] for col in pii_columns]
        raise ValueError(
            f"PII detected in DataFrame columns: {', '.join(pii_columns)}. "
            f"Entity types found: {', '.join(entity_types)}."
        )

    return True
