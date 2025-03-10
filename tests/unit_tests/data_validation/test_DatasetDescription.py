import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.DatasetDescription import DatasetDescription
from validmind import RawData


class TestDatasetDescription(unittest.TestCase):
    def setUp(self):
        # Create a dataset with different types of columns
        df = pd.DataFrame(
            {
                "numeric": pd.Series(
                    [1.5, 2.5, 3.5, np.nan, 5.5, 6.6, 7.7],
                    dtype=np.float64,
                ),  # More values, more decimals
                "categorical": pd.Categorical(
                    ["A", "B", "A", "C", "B", "C", "A"]
                ),  # Explicitly categorical
                "boolean": pd.Series(
                    [True, False, True, True, False, True, False], dtype=bool
                ),  # Explicitly boolean
                "text": [
                    "hello@gmail.com",
                    "this is a longer text",
                    "hello world",
                    "this is a longer text",
                    "this is a longer text",
                    "another example of text",
                    "this is a longer text",
                ],  # Text
                "all_null": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],  # All null column
                "binary_numeric": [0, 1, 0, 1, 0, 1, 0],  # Added binary numeric column
                "binary_float": [
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                ],  # Added binary float column
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            target_column=None,
            __log=False,
        )

    def test_returns_expected_structure(self):
        result = DatasetDescription(self.vm_dataset)

        # Check if result is a tuple with expected structure (Dataset Description, RawData)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        description, raw_data = result

        # Check if description is a dictionary with expected key
        self.assertIsInstance(description, dict)
        self.assertIn("Dataset Description", description)

        # Check if description is a list of dictionaries
        description_list = description["Dataset Description"]
        self.assertIsInstance(description_list, list)
        self.assertTrue(all(isinstance(item, dict) for item in description_list))

        # Check if each column description has required fields
        # Note: Count is not included as it's not available for Null type columns
        required_fields = [
            "Name",
            "Type",
            "Count",
            "Missing",
            "Missing %",
            "Distinct",
            "Distinct %",
        ]
        for item in description_list:
            for field in required_fields:
                self.assertIn(field, item)
                self.assertIsNotNone(item[field])

        # Check Count field separately as it's not available for Null columns
        for item in description_list:
            if item["Type"] != "Null":
                self.assertIn("Count", item)
                self.assertIsNotNone(item["Count"])

        # Check raw_data is instance of RawData
        self.assertIsInstance(raw_data, RawData)

    def test_column_types_and_stats(self):
        result = DatasetDescription(self.vm_dataset)
        print(self.vm_dataset.df.head())
        print(self.vm_dataset.df.dtypes)
        description, _ = result
        column_info = {
            item["Name"]: item for item in description["Dataset Description"]
        }

        # Check numeric column
        self.assertEqual(column_info["numeric"]["Type"], "Numeric")
        self.assertEqual(column_info["numeric"]["Missing"], 1)  # One NaN
        self.assertEqual(column_info["numeric"]["Distinct"], 6)  # 6 unique values
        self.assertEqual(column_info["numeric"]["Count"], 6)  # 6 total

        # Check categorical column
        self.assertEqual(column_info["categorical"]["Type"], "Categorical")
        self.assertEqual(column_info["categorical"]["Distinct"], 3)  # A, B, C
        self.assertEqual(column_info["categorical"]["Missing"], 0)  # No missing values
        self.assertEqual(column_info["categorical"]["Count"], 7)  # All present

        # Check boolean column - should be treated as categorical
        self.assertEqual(column_info["boolean"]["Type"], "Categorical")
        self.assertEqual(column_info["boolean"]["Distinct"], 2)  # True, False
        self.assertEqual(column_info["boolean"]["Missing"], 0)  # No missing values
        self.assertEqual(column_info["boolean"]["Count"], 7)  # All present

        # Check text column
        print(column_info)
        self.assertEqual(column_info["text"]["Type"], "Text")
        self.assertEqual(column_info["text"]["Distinct"],4)  # 4 unique strings
        self.assertEqual(column_info["text"]["Missing"], 0)  # No missing values
        self.assertEqual(column_info["text"]["Count"], 7)  # All present

        # Check null column
        self.assertEqual(column_info["all_null"]["Type"], "Null")
        self.assertEqual(column_info["all_null"]["Missing"], 7)  # All values missing
        # Note: Count is not checked for Null type columns

        # Check binary numeric columns
        self.assertEqual(
            column_info["binary_numeric"]["Type"], "Categorical"
        )  # ydata_profiling infers this as categorical
        self.assertEqual(column_info["binary_numeric"]["Distinct"], 2)  # 0 and 1
        self.assertEqual(column_info["binary_numeric"]["Missing"], 0)
        self.assertEqual(column_info["binary_numeric"]["Count"], 7)

        # Check binary float columns
        self.assertEqual(
            column_info["binary_float"]["Type"], "Categorical"
        )  # ydata_profiling infers this as categorical
        self.assertEqual(column_info["binary_float"]["Distinct"], 2)  # 0.0 and 1.0
        self.assertEqual(column_info["binary_float"]["Missing"], 0)
        self.assertEqual(column_info["binary_float"]["Count"], 7)
