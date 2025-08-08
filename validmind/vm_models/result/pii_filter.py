# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
PII filtering utilities using Microsoft Presidio for detecting and masking
personally identifiable information in test result data.
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Union

import pandas as pd

from ...logging import get_logger

logger = get_logger(__name__)


class PIIDetectionMode(Enum):
    """Enum for PII detection modes."""

    DISABLED = "disabled"
    TEST_RESULTS = "test_results"
    TEST_DESCRIPTIONS = "test_descriptions"
    ALL = "all"


def _get_pii_detection_mode() -> PIIDetectionMode:
    """Get the current PII detection mode from environment variable."""
    mode_str = os.getenv("VALIDMIND_PII_DETECTION", "disabled").lower()

    try:
        return PIIDetectionMode(mode_str)
    except ValueError:
        logger.warning(
            f"Invalid PII detection mode '{mode_str}'. "
            f"Valid options: {', '.join([mode.value for mode in PIIDetectionMode])}. "
            f"Defaulting to 'disabled'."
        )
        return PIIDetectionMode.DISABLED


# Lazy load presidio components to avoid import errors when not installed
_analyzer = None


def _get_presidio_analyzer():
    """Lazy load Presidio analyzer to avoid import errors when not installed."""
    global _analyzer
    if _analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore

            _analyzer = AnalyzerEngine()
            logger.debug("Presidio analyzer initialized successfully")
        except ImportError:
            logger.warning(
                "Presidio analyzer not available. Install with: pip install validmind[pii-detection]"
            )
            _analyzer = False
    return _analyzer if _analyzer is not False else None


def is_pii_detection_enabled_for_test_results() -> bool:
    """Check if PII detection is enabled for test results and available."""
    mode = _get_pii_detection_mode()
    return (
        mode in [PIIDetectionMode.TEST_RESULTS, PIIDetectionMode.ALL]
        and _get_presidio_analyzer() is not None
    )


def is_pii_detection_enabled_for_test_descriptions() -> bool:
    """Check if PII detection is enabled for test descriptions and available."""
    mode = _get_pii_detection_mode()
    return (
        mode in [PIIDetectionMode.TEST_DESCRIPTIONS, PIIDetectionMode.ALL]
        and _get_presidio_analyzer() is not None
    )


def is_pii_detection_enabled() -> bool:
    """Check if PII detection is enabled for any mode and available."""
    mode = _get_pii_detection_mode()
    return mode != PIIDetectionMode.DISABLED and _get_presidio_analyzer() is not None


def detect_pii_in_text(
    text: str,
    entities: Optional[List[str]] = None,
    language: str = "en",
    threshold: float = 0.5,
) -> List[Dict]:
    """
    Detect PII entities in text using Presidio analyzer.

    Args:
        text: The text to analyze for PII
        entities: List of entity types to detect. If None, detects all supported entities
        language: Language code for analysis (default: "en")
        threshold: Minimum confidence score for PII detection (default: 0.5)

    Returns:
        List of detected PII entities with their positions and confidence scores
    """
    analyzer = _get_presidio_analyzer()
    if analyzer is None:
        logger.debug("PII detection skipped - Presidio not available")
        return []

    try:
        # Default entities to detect common PII types
        if entities is None:
            entities = [
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

        results = analyzer.analyze(text=text, entities=entities, language=language)

        # Filter results by confidence threshold
        filtered_results = [
            {
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": result.score,
                "text": text[result.start : result.end],
            }
            for result in results
            if result.score >= threshold
        ]

        if filtered_results:
            logger.debug(f"Detected {len(filtered_results)} PII entities in text")

        return filtered_results

    except Exception as e:
        logger.warning(f"PII detection failed: {e}")
        return []


def scan_dataframe_for_pii(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.5,
    sample_size: int = 100,
) -> Dict[str, List[Dict]]:
    """
    Scan a pandas DataFrame for PII content in text columns.

    Args:
        df: The DataFrame to scan
        columns: List of column names to scan (if None, scans all string columns)
        threshold: Minimum confidence score for PII detection
        sample_size: Maximum number of rows to sample for PII detection

    Returns:
        Dictionary mapping column names to lists of detected PII entities
    """
    if not is_pii_detection_enabled_for_test_results():
        return {}

    pii_findings = {}

    # Determine which columns to scan
    if columns is None:
        # Scan all string/object columns
        columns = [col for col in df.columns if df[col].dtype == "object"]

    # Limit the number of rows to scan for performance
    sample_df = df.head(sample_size) if len(df) > sample_size else df

    for column in columns:
        column_pii = []

        # Scan non-null string values in the column
        for idx, value in sample_df[column].dropna().items():
            if isinstance(value, str) and len(value.strip()) > 0:
                pii_entities = detect_pii_in_text(text=str(value), threshold=threshold)

                if pii_entities:
                    column_pii.extend(
                        [
                            {**entity, "row_index": idx, "column": column}
                            for entity in pii_entities
                        ]
                    )

        if column_pii:
            pii_findings[column] = column_pii
            logger.info(f"Found {len(column_pii)} PII entities in column '{column}'")

    return pii_findings


def check_table_for_pii(
    table_data: Union[pd.DataFrame, List[Dict]],
    threshold: float = 0.5,
    raise_on_detection: bool = True,
) -> None:
    """
    Check a table (DataFrame or list of dicts) for PII content.

    Args:
        table_data: The table data to check
        threshold: Minimum confidence score for PII detection
        raise_on_detection: If True, raises ValueError when PII is detected (default: True)

    Raises:
        ValueError: If PII is detected and raise_on_detection is True
    """
    if not is_pii_detection_enabled_for_test_results():
        return

    # Convert to DataFrame if it's a list of dicts
    if isinstance(table_data, list):
        if not table_data:
            return
        df = pd.DataFrame(table_data)
    else:
        df = table_data

    # Scan for PII
    pii_findings = scan_dataframe_for_pii(df, threshold=threshold)
    has_pii = bool(pii_findings)

    if has_pii and raise_on_detection:
        entity_types = set()
        for findings in pii_findings.values():
            entity_types.update(entity["entity_type"] for entity in findings)

        raise ValueError(
            f"PII detected in table data. Entity types found: {', '.join(entity_types)}. "
            f"Pass `unsafe=True` to bypass PII detection."
        )


def check_text_for_pii(
    text: str,
    entities: Optional[List[str]] = None,
    language: str = "en",
    threshold: float = 0.5,
    raise_on_detection: bool = True,
) -> List[Dict]:
    """
    Check text for PII content and optionally raise an exception.

    Args:
        text: The text to check for PII
        entities: List of entity types to detect
        language: Language code for analysis
        threshold: Minimum confidence score for PII detection
        raise_on_detection: If True, raises ValueError when PII is detected (default: True)

    Returns:
        List of detected PII entities

    Raises:
        ValueError: If PII is detected and raise_on_detection is True
    """
    if not is_pii_detection_enabled_for_test_descriptions():
        return []

    pii_entities = detect_pii_in_text(
        text=text, entities=entities, language=language, threshold=threshold
    )

    if pii_entities and raise_on_detection:
        entity_types = set(entity["entity_type"] for entity in pii_entities)
        raise ValueError(
            f"PII detected in text content. Entity types found: {', '.join(entity_types)}. "
            f"Pass `unsafe=True` to bypass PII detection."
        )

    return pii_entities
