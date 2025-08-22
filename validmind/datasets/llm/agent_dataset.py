# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
LLM Agent Dataset for integrating with DeepEval evaluation framework.

This module provides an LLMAgentDataset class that inherits from VMDataset
and enables the use of all DeepEval tests and metrics within the ValidMind library.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from validmind.logging import get_logger
from validmind.vm_models.dataset import VMDataset

logger = get_logger(__name__)

# Optional DeepEval imports with graceful fallback
try:
    from deepeval import evaluate
    from deepeval.dataset import EvaluationDataset, Golden
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase, ToolCall

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
        input_id: str = None,
        test_cases: Optional[List] = None,
        goldens: Optional[List] = None,
        deepeval_dataset: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize LLMAgentDataset.

        Args:
            input_id: Identifier for the dataset
            test_cases: List of DeepEval LLMTestCase objects
            goldens: List of DeepEval Golden objects
            deepeval_dataset: DeepEval EvaluationDataset instance
            **kwargs: Additional arguments passed to VMDataset
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
        """Convert DeepEval test cases/goldens to pandas DataFrame."""
        data = []

        # Process test cases
        for i, test_case in enumerate(self.test_cases):
            row = {
                "id": f"test_case_{i}",
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "expected_output": getattr(test_case, "expected_output", None),
                "context": self._serialize_list_field(
                    getattr(test_case, "context", None)
                ),
                "retrieval_context": self._serialize_list_field(
                    getattr(test_case, "retrieval_context", None)
                ),
                "tools_called": self._serialize_tools_field(
                    getattr(test_case, "tools_called", None)
                ),
                "expected_tools": self._serialize_tools_field(
                    getattr(test_case, "expected_tools", None)
                ),
                "type": "test_case",
            }
            data.append(row)

        # Process goldens
        for i, golden in enumerate(self.goldens):
            row = {
                "id": f"golden_{i}",
                "input": golden.input,
                "actual_output": getattr(golden, "actual_output", None),
                "expected_output": getattr(golden, "expected_output", None),
                "context": self._serialize_list_field(getattr(golden, "context", None)),
                "retrieval_context": self._serialize_list_field(
                    getattr(golden, "retrieval_context", None)
                ),
                "tools_called": self._serialize_tools_field(
                    getattr(golden, "tools_called", None)
                ),
                "expected_tools": self._serialize_tools_field(
                    getattr(golden, "expected_tools", None)
                ),
                "type": "golden",
            }
            data.append(row)

        if not data:
            # Create empty DataFrame with expected columns
            data = [
                {
                    "id": "",
                    "input": "",
                    "actual_output": "",
                    "expected_output": "",
                    "context": "",
                    "retrieval_context": "",
                    "tools_called": "",
                    "expected_tools": "",
                    "type": "",
                }
            ]

        return pd.DataFrame(data)

    def _serialize_list_field(self, field: Optional[List[str]]) -> str:
        """Serialize list field to string for DataFrame storage."""
        if field is None:
            return ""
        return "|".join(str(item) for item in field)

    def _serialize_tools_field(self, tools: Optional[List]) -> str:
        """Serialize tools list to string for DataFrame storage."""
        if tools is None:
            return ""
        tool_strs = []
        for tool in tools:
            if hasattr(tool, "name"):
                tool_strs.append(tool.name)
            else:
                tool_strs.append(str(tool))
        return "|".join(tool_strs)

    def _deserialize_list_field(self, field_str: str) -> List[str]:
        """Deserialize string back to list."""
        if not field_str:
            return []
        return field_str.split("|")

    @classmethod
    def from_test_cases(
        cls, test_cases: List, input_id: str = "llm_agent_dataset", **kwargs
    ) -> "LLMAgentDataset":
        """
        Create LLMAgentDataset from DeepEval test cases.

        Args:
            test_cases: List of DeepEval LLMTestCase objects
            input_id: Dataset identifier
            **kwargs: Additional arguments

        Returns:
            LLMAgentDataset instance
        """
        return cls(input_id=input_id, test_cases=test_cases, **kwargs)

    @classmethod
    def from_goldens(
        cls, goldens: List, input_id: str = "llm_agent_dataset", **kwargs
    ) -> "LLMAgentDataset":
        """
        Create LLMAgentDataset from DeepEval goldens.

        Args:
            goldens: List of DeepEval Golden objects
            input_id: Dataset identifier
            **kwargs: Additional arguments

        Returns:
            LLMAgentDataset instance
        """
        return cls(input_id=input_id, goldens=goldens, **kwargs)

    @classmethod
    def from_deepeval_dataset(
        cls, deepeval_dataset, input_id: str = "llm_agent_dataset", **kwargs
    ) -> "LLMAgentDataset":
        """
        Create LLMAgentDataset from DeepEval EvaluationDataset.

        Args:
            deepeval_dataset: DeepEval EvaluationDataset instance
            input_id: Dataset identifier
            **kwargs: Additional arguments

        Returns:
            LLMAgentDataset instance
        """
        return cls(
            input_id=input_id,
            test_cases=getattr(deepeval_dataset, "test_cases", []),
            goldens=getattr(deepeval_dataset, "goldens", []),
            deepeval_dataset=deepeval_dataset,
            **kwargs,
        )

    def add_test_case(self, test_case) -> None:
        """
        Add a DeepEval test case to the dataset.

        Args:
            test_case: DeepEval LLMTestCase instance
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval is required to add test cases")

        self.test_cases.append(test_case)
        # Refresh the DataFrame
        df = self._convert_to_dataframe()
        self._df = df
        self.columns = df.columns.tolist()

    def add_golden(self, golden) -> None:
        """
        Add a DeepEval golden to the dataset.

        Args:
            golden: DeepEval Golden instance
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval is required to add goldens")

        self.goldens.append(golden)
        # Refresh the DataFrame
        df = self._convert_to_dataframe()
        self._df = df
        self.columns = df.columns.tolist()

    def convert_goldens_to_test_cases(self, llm_app_function) -> None:
        """
        Convert goldens to test cases by generating actual outputs.

        Args:
            llm_app_function: Function that takes input and returns LLM output
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

    def evaluate_with_deepeval(self, metrics: List, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the dataset using DeepEval metrics.

        Args:
            metrics: List of DeepEval metric instances
            **kwargs: Additional arguments passed to deepeval.evaluate()

        Returns:
            Evaluation results dictionary
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

    def get_deepeval_dataset(self):
        """
        Get or create a DeepEval EvaluationDataset instance.

        Returns:
            DeepEval EvaluationDataset instance
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

    def to_deepeval_test_cases(self) -> List:
        """
        Convert dataset rows back to DeepEval test cases.

        Returns:
            List of DeepEval LLMTestCase objects
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
                        context=self._deserialize_list_field(context_val)
                        if context_val
                        else None,
                        retrieval_context=self._deserialize_list_field(
                            retrieval_context_val
                        )
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
