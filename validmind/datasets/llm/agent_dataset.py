# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
LLM Agent Dataset for integrating with DeepEval evaluation framework.

This module provides an LLMAgentDataset class that inherits from VMDataset
and enables the use of all DeepEval tests and metrics within the ValidMind library.
"""

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from validmind.logging import get_logger
from validmind.vm_models.dataset import VMDataset

logger = get_logger(__name__)

# Optional DeepEval imports with graceful fallback
try:
    from deepeval import evaluate
    from deepeval.dataset import EvaluationDataset
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    LLMTestCase = None
    ToolCall = None
    EvaluationDataset = None
    Golden = None
    BaseMetric = None
    evaluate = None


class LLMAgentDataset(VMDataset):
    """
    LLM Agent Dataset for DeepEval integration with ValidMind.

    This dataset class allows you to use all DeepEval tests and metrics
    within the ValidMind evaluation framework. It stores LLM interaction data
    in a format compatible with both frameworks.

    Attributes:
        test_cases (List[LLMTestCase]): List of DeepEval test cases
        goldens (List[Golden]): List of DeepEval golden templates
        deepeval_dataset (EvaluationDataset): DeepEval dataset instance

    Example:
        ```python
        # Create from DeepEval test cases
        test_cases = [
            LLMTestCase(
                input="What is machine learning?",
                actual_output="Machine learning is a subset of AI...",
                expected_output="ML is a method of data analysis...",
                context=["Machine learning context..."]
            )
        ]

        dataset = LLMAgentDataset.from_test_cases(
            test_cases=test_cases,
            input_id="llm_eval_dataset"
        )

        # Run DeepEval metrics
        from deepeval.metrics import AnswerRelevancyMetric
        results = dataset.evaluate_with_deepeval([AnswerRelevancyMetric()])
        ```
    """

    def __init__(
        self,
        input_id: Optional[str] = None,
        test_cases: Optional[List[Any]] = None,
        goldens: Optional[List[Any]] = None,
        deepeval_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LLMAgentDataset.

        Args:
            input_id (Optional[str]): Identifier for the dataset.
            test_cases (Optional[List[LLMTestCase]]): List of DeepEval LLMTestCase objects.
            goldens (Optional[List[Golden]]): List of DeepEval Golden objects.
            deepeval_dataset (Optional[EvaluationDataset]): DeepEval EvaluationDataset instance.
            **kwargs (Any): Additional arguments passed to `VMDataset`.
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "DeepEval is required to use LLMAgentDataset. "
                "Install it with: pip install deepeval"
            )

        # Store DeepEval objects
        self.test_cases = test_cases or []
        self.goldens = goldens or []
        self.deepeval_dataset = deepeval_dataset

        # Convert to pandas DataFrame for VMDataset compatibility
        df = self._convert_to_dataframe()

        # Initialize VMDataset with the converted data
        super().__init__(
            raw_dataset=df.values,
            input_id=input_id or "llm_agent_dataset",
            columns=df.columns.tolist(),
            text_column="input",  # The input text for LLM
            target_column="expected_output",  # Expected response
            extra_columns={
                "actual_output": "actual_output",
                "context": "context",
                "retrieval_context": "retrieval_context",
                "tools_called": "tools_called",
                "expected_tools": "expected_tools",
            },
            **kwargs,
        )

    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Convert DeepEval test cases/goldens to pandas DataFrame.

        Returns:
            pandas.DataFrame: Tabular representation of test cases and goldens.
        """
        data = []
        data.extend(self._process_test_cases())
        data.extend(self._process_goldens())

        if not data:
            data = [self._get_empty_row()]

        return pd.DataFrame(data)

    def _process_test_cases(self) -> List[Dict[str, Any]]:
        """Process test cases into DataFrame rows."""
        data = []
        for i, test_case in enumerate(self.test_cases):
            row = {
                "id": f"test_case_{i}",
                "input": test_case.input,
                "actual_output": test_case.actual_output,
            }
            self._add_optional_fields(row, test_case)
            data.append(row)
        return data

    def _process_goldens(self) -> List[Dict[str, Any]]:
        """Process goldens into DataFrame rows."""
        data = []
        for i, golden in enumerate(self.goldens):
            row = {"id": f"golden_{i}", "input": golden.input}
            self._add_optional_fields(row, golden)
            data.append(row)
        return data

    def _add_optional_fields(self, row: Dict[str, Any], obj: Any) -> None:
        """Add optional fields to a row from an object."""
        optional_fields = [
            "expected_output",
            "context",
            "retrieval_context",
            "tools_called",
            "expected_tools",
        ]
        for field in optional_fields:
            value = getattr(obj, field, None)
            if value is not None:
                row[field] = value

    def _get_empty_row(self) -> Dict[str, str]:
        """Get an empty row with all expected columns."""
        return {
            "id": "",
            "input": "",
            "actual_output": "",
            "expected_output": "",
            "context": "",
            "retrieval_context": "",
            "tools_called": "",
            "expected_tools": "",
        }

    @classmethod
    def from_test_cases(
        cls, test_cases: List[Any], input_id: str = "llm_agent_dataset", **kwargs: Any
    ) -> "LLMAgentDataset":
        """
        Create LLMAgentDataset from DeepEval test cases.

        Args:
            test_cases (List[LLMTestCase]): List of DeepEval LLMTestCase objects.
            input_id (str): Dataset identifier.
            **kwargs (Any): Additional arguments passed through to constructor.

        Returns:
            LLMAgentDataset: New dataset instance.
        """
        return cls(input_id=input_id, test_cases=test_cases, **kwargs)

    @classmethod
    def from_goldens(
        cls, goldens: List[Any], input_id: str = "llm_agent_dataset", **kwargs: Any
    ) -> "LLMAgentDataset":
        """
        Create LLMAgentDataset from DeepEval goldens.

        Args:
            goldens (List[Golden]): List of DeepEval Golden objects.
            input_id (str): Dataset identifier.
            **kwargs (Any): Additional arguments passed through to constructor.

        Returns:
            LLMAgentDataset: New dataset instance.
        """
        return cls(input_id=input_id, goldens=goldens, **kwargs)

    @classmethod
    def from_deepeval_dataset(
        cls, deepeval_dataset: Any, input_id: str = "llm_agent_dataset", **kwargs: Any
    ) -> "LLMAgentDataset":
        """
        Create LLMAgentDataset from DeepEval EvaluationDataset.

        Args:
            deepeval_dataset (EvaluationDataset): DeepEval EvaluationDataset instance.
            input_id (str): Dataset identifier.
            **kwargs (Any): Additional arguments passed through to constructor.

        Returns:
            LLMAgentDataset: New dataset instance.
        """
        return cls(
            input_id=input_id,
            test_cases=getattr(deepeval_dataset, "test_cases", []),
            goldens=getattr(deepeval_dataset, "goldens", []),
            deepeval_dataset=deepeval_dataset,
            **kwargs,
        )

    def add_test_case(self, test_case: Any) -> None:
        """
        Add a DeepEval test case to the dataset.

        Args:
            test_case (LLMTestCase): DeepEval LLMTestCase instance.
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval is required to add test cases")

        self.test_cases.append(test_case)
        # Refresh the DataFrame
        df = self._convert_to_dataframe()
        self._df = df
        self.columns = df.columns.tolist()

    def add_golden(self, golden: Any) -> None:
        """
        Add a DeepEval golden to the dataset.

        Args:
            golden (Golden): DeepEval Golden instance.
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval is required to add goldens")

        self.goldens.append(golden)
        # Refresh the DataFrame
        df = self._convert_to_dataframe()
        self._df = df
        self.columns = df.columns.tolist()

    def convert_goldens_to_test_cases(
        self, llm_app_function: Callable[[str], Any]
    ) -> None:
        """
        Convert goldens to test cases by generating actual outputs.

        Args:
            llm_app_function (Callable[[str], Any]): Function that takes input and returns LLM output.
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval is required for conversion")

        new_test_cases = []
        for golden in self.goldens:
            try:
                actual_output = llm_app_function(golden.input)
                if LLMTestCase is not None:
                    test_case = LLMTestCase(
                        input=golden.input,
                        actual_output=actual_output,
                        expected_output=getattr(golden, "expected_output", None),
                        context=getattr(golden, "context", None),
                        retrieval_context=getattr(golden, "retrieval_context", None),
                        tools_called=getattr(golden, "tools_called", None),
                        expected_tools=getattr(golden, "expected_tools", None),
                    )
                else:
                    raise ImportError("DeepEval LLMTestCase is not available")
                new_test_cases.append(test_case)
            except Exception as e:
                logger.warning(f"Failed to convert golden to test case: {e}")
                continue

        self.test_cases.extend(new_test_cases)
        # Refresh the DataFrame
        df = self._convert_to_dataframe()
        self._df = df
        self.columns = df.columns.tolist()

    def evaluate_with_deepeval(
        self, metrics: List[Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Evaluate the dataset using DeepEval metrics.

        Args:
            metrics (List[Any]): List of DeepEval metric instances.
            **kwargs (Any): Additional arguments passed to `deepeval.evaluate()`.

        Returns:
            Dict[str, Any]: Evaluation results dictionary.
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval is required for evaluation")

        if not self.test_cases:
            raise ValueError("No test cases available for evaluation")

        try:
            # Use DeepEval's evaluate function
            if evaluate is not None:
                results = evaluate(
                    test_cases=self.test_cases, metrics=metrics, **kwargs
                )
                return results
            else:
                raise ImportError("DeepEval evaluate function is not available")
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            raise

    def get_deepeval_dataset(self) -> Any:
        """
        Get or create a DeepEval EvaluationDataset instance.

        Returns:
            EvaluationDataset: DeepEval EvaluationDataset instance.
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval is required to get dataset")

        if self.deepeval_dataset is None:
            if EvaluationDataset is not None:
                self.deepeval_dataset = EvaluationDataset(goldens=self.goldens)
                # Add test cases if available
                for test_case in self.test_cases:
                    self.deepeval_dataset.add_test_case(test_case)
            else:
                raise ImportError("DeepEval EvaluationDataset is not available")

        return self.deepeval_dataset

    def to_deepeval_test_cases(self) -> List[Any]:
        """
        Convert dataset rows back to DeepEval test cases.

        Returns:
            List[LLMTestCase]: List of DeepEval LLMTestCase objects.
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval is required for conversion")

        test_cases = []
        for _, row in self.df.iterrows():
            # Check if this row has actual output (is a test case)
            has_actual_output = (
                pd.notna(row["actual_output"])
                and str(row["actual_output"]).strip() != ""
            )
            is_test_case = str(row["type"]) == "test_case"

            if is_test_case or has_actual_output:
                if LLMTestCase is not None:
                    # Safely get context fields
                    context_val = (
                        row["context"]
                        if pd.notna(row["context"]) and str(row["context"]).strip()
                        else None
                    )
                    retrieval_context_val = (
                        row["retrieval_context"]
                        if pd.notna(row["retrieval_context"])
                        and str(row["retrieval_context"]).strip()
                        else None
                    )
                    expected_output_val = (
                        row["expected_output"]
                        if pd.notna(row["expected_output"])
                        and str(row["expected_output"]).strip()
                        else None
                    )

                    test_case = LLMTestCase(
                        input=str(row["input"]),
                        actual_output=str(row["actual_output"])
                        if pd.notna(row["actual_output"])
                        else "",
                        expected_output=expected_output_val,
                        context=context_val if context_val else None,
                        retrieval_context=retrieval_context_val
                        if retrieval_context_val
                        else None,
                        # Note: tools_called deserialization would need more complex logic
                        # for now we'll keep it simple
                    )
                    test_cases.append(test_case)
                else:
                    raise ImportError("DeepEval LLMTestCase is not available")

        return test_cases

    def __repr__(self) -> str:
        return (
            f"LLMAgentDataset(input_id='{self.input_id}', "
            f"test_cases={len(self.test_cases)}, "
            f"goldens={len(self.goldens)})"
        )
