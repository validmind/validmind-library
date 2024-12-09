import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import validmind as vm
from validmind.tests.model_validation.ModelMetadata import ModelMetadata
from validmind.errors import UnsupportedModelError


class TestModelMetadata(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create and train a simple linear regression model
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = LinearRegression()
        model.fit(X, y)

        # Initialize ValidMind model with LinearRegression
        self.vm_model = vm.init_model(
            input_id="linear_regression",
            model=model,
            __log=False,
        )

    def test_returns_dataframe(self):
        """Test if function returns a pandas DataFrame with expected structure."""
        result = ModelMetadata(self.vm_model)

        # Check return type
        self.assertIsInstance(result, pd.DataFrame)

        # Check expected columns
        expected_columns = [
            "Modeling Technique",
            "Modeling Framework",
            "Framework Version",
            "Programming Language",
        ]
        self.assertListEqual(list(result.columns), expected_columns)

    def test_metadata_values(self):
        """Test if metadata values are correctly extracted and labeled."""
        result_df = ModelMetadata(self.vm_model)

        # Check first row values
        row = result_df.iloc[0]
        self.assertEqual(row["Modeling Technique"], "SKlearnModel")
        self.assertEqual(row["Modeling Framework"], "sklearn")
        self.assertIsNotNone(row["Framework Version"])
        self.assertEqual(row["Programming Language"], "Python")

        # Verify 'params' is not included
        self.assertNotIn("params", result_df.columns)

    def test_empty_model_metadata(self):
        """Test metadata for a model with minimal metadata."""
        vm_model_empty = vm.init_model(
            input_id="empty_model",
            attributes={"architecture": "PyMC", "language": "Python"},
            __log=False,
        )

        result_df = ModelMetadata(vm_model_empty)

        # Check metadata values
        row = result_df.iloc[0]
        self.assertEqual(row["Modeling Technique"], "MetadataModel")
        self.assertEqual(row["Modeling Framework"], "MetadataModel")
        self.assertEqual(row["Framework Version"], "N/A")
        self.assertEqual(row["Programming Language"], "Python")

    def test_missing_metadata(self):
        """Test that appropriate error is raised when required metadata fields are missing."""
        with self.assertRaises(UnsupportedModelError) as context:
            _ = vm.init_model(
                input_id="incomplete_model",
                attributes={
                    "language": "Python"
                    # architecture intentionally omitted
                },
                __log=False,
            )

        # Check the error message
        self.assertIn("Model attributes", str(context.exception))
        self.assertIn("missing required keys", str(context.exception))
        self.assertIn("architecture", str(context.exception))
