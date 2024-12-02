import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.HighCardinality import HighCardinality


class TestHighCardinality(unittest.TestCase):
    def setUp(self):
        # Create a dataset with different cardinality levels
        df = pd.DataFrame(
            {
                "low_cardinality": ["A", "B", "C", "A", "B", "C"] * 10,  # 60 entries
                "high_cardinality": [f"val_{i}" for i in range(60)],  # 60 unique values
                "numeric": range(60),  # 60 numeric values
            }
        )

        # Initialize VMDataset without specifying feature_columns_categorical
        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            __log=False,
        )

    def test_cardinality_structure(self):
        results, all_passed = HighCardinality(self.vm_dataset)

        # Check basic structure
        self.assertIsInstance(results, list)
        self.assertIsInstance(all_passed, bool)

        # Check that results include both columns
        column_names = [result["Column"] for result in results]
        self.assertIn("low_cardinality", column_names)
        self.assertIn("high_cardinality", column_names)

    def test_cardinality_values(self):
        results, _ = HighCardinality(self.vm_dataset)

        # Convert results to dictionary for easier testing
        results_dict = {result["Column"]: result for result in results}

        # Test low cardinality column
        low_card = results_dict["low_cardinality"]
        self.assertEqual(low_card["Number of Distinct Values"], 3)
        self.assertAlmostEqual(
            low_card["Percentage of Distinct Values (%)"], 5.0
        )  # 3/60 * 100
        self.assertEqual(low_card["Pass/Fail"], "Pass")

        # Test high cardinality column
        high_card = results_dict["high_cardinality"]
        self.assertEqual(high_card["Number of Distinct Values"], 60)
        self.assertAlmostEqual(
            high_card["Percentage of Distinct Values (%)"], 100.0
        )  # 60/60 * 100
        self.assertEqual(
            high_card["Pass/Fail"], "Fail"
        )  # Assuming default threshold of 10%

    def test_all_passed_flag(self):
        # Default thresholds should result in not all passing
        _, all_passed_default = HighCardinality(self.vm_dataset)

        # Only test the default case
        self.assertFalse(all_passed_default)

    def test_numeric_columns_ignored(self):
        results, _ = HighCardinality(self.vm_dataset)
        column_names = [result["Column"] for result in results]

        # Numeric column should not be in results
        self.assertNotIn("numeric", column_names)
