import unittest
import pandas as pd
import validmind as vm
from validmind.errors import SkipTestError
from validmind.tests.data_validation.WOEBinTable import WOEBinTable, RawData


class TestWOEBinTable(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset with categorical and numeric features and binary target
        n_samples = 1000

        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"] * (n_samples // 3) + ["A"] * (n_samples % 3),
                "cat2": ["X", "Y"] * (n_samples // 2) + ["X"] * (n_samples % 2),
                "numeric": range(n_samples),
                "target": [0, 1] * (n_samples // 2)
                + [0] * (n_samples % 2),  # Binary target
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            feature_columns=["cat1", "cat2", "numeric"],
            target_column="target",
            __log=False,
        )

        # Create dataset with no target column
        df_no_target = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"] * (n_samples // 3) + ["A"] * (n_samples % 3),
                "cat2": ["X", "Y"] * (n_samples // 2) + ["X"] * (n_samples % 2),
            }
        )

        self.vm_dataset_no_target = vm.init_dataset(
            input_id="test_dataset_no_target",
            dataset=df_no_target,
            feature_columns=["cat1", "cat2"],
            __log=False,
        )

    def test_woe_bin_table(self):
        result, raw_data = WOEBinTable(self.vm_dataset)

        # Check the table structure
        table = result["Weight of Evidence (WoE) and Information Value (IV)"]
        self.assertIsInstance(table, pd.DataFrame)

        # Check required columns are present
        required_columns = ["variable", "bin_number"]
        self.assertTrue(all(col in table.columns for col in required_columns))

        # Check that we have entries for all features
        unique_variables = table["variable"].unique()
        self.assertEqual(
            len(unique_variables), 3
        )  # Should have entries for all 3 features
        expected_features = ["cat1", "cat2", "numeric"]
        self.assertTrue(all(feat in unique_variables for feat in expected_features))

        # Check that raw data is an instance of RawData
        self.assertIsInstance(raw_data, RawData)

    def test_no_target(self):
        # Should raise SkipTestError when no target column present
        with self.assertRaises(SkipTestError):
            WOEBinTable(self.vm_dataset_no_target)
