import unittest
import numpy as np
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.TabularDescriptionTables import (
    TabularDescriptionTables,
)


class TestTabularDescriptionTables(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with different types of columns
        np.random.seed(42)  # For reproducibility

        # Create sample data
        self.df = pd.DataFrame(
            {
                # Numerical columns
                "num1": [1, 2, 3, 4, 5],
                "num2": [1.1, 2.2, np.nan, 4.4, 5.5],  # Including missing values
                # Categorical columns
                "cat1": ["A", "B", "A", "C", "B"],
                "cat2": ["X", np.nan, "X", "Y", "Y"],  # Including missing values
                # Datetime columns
                "date1": pd.date_range("2023-01-01", periods=5),
                "date2": [
                    pd.NaT,
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],  # Including missing values
            }
        )

        # Convert date2 to datetime
        self.df["date2"] = pd.to_datetime(self.df["date2"])

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            __log=False,
        )

    def test_returns_tuple_of_dataframes(self):
        # Run the function
        result = TabularDescriptionTables(self.vm_dataset)

        # Check if result is a tuple
        self.assertIsInstance(result, tuple)

        # Check if all elements in the tuple are DataFrames
        for df in result:
            self.assertIsInstance(df, pd.DataFrame)

    def test_correct_number_of_tables(self):
        # Run the function
        result = TabularDescriptionTables(self.vm_dataset)

        # Should return 3 tables (numerical, categorical, datetime)
        self.assertEqual(len(result), 3)

    def test_table_contents(self):
        result = TabularDescriptionTables(self.vm_dataset)

        # Test numerical table
        numerical_table = next(
            df for df in result if "Numerical Variable" in df.columns
        )
        self.assertEqual(len(numerical_table), 2)  # Should have 2 numerical columns
        self.assertTrue("Missing Values (%)" in numerical_table.columns)

        # Test categorical table
        categorical_table = next(
            df for df in result if "Categorical Variable" in df.columns
        )
        self.assertEqual(len(categorical_table), 2)  # Should have 2 categorical columns
        self.assertTrue("Num of Unique Values" in categorical_table.columns)

        # Test datetime table
        datetime_table = next(df for df in result if "Datetime Variable" in df.columns)
        self.assertEqual(len(datetime_table), 2)  # Should have 2 datetime columns
        self.assertTrue("Earliest Date" in datetime_table.columns)
