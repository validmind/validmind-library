import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.DescriptiveStatistics import DescriptiveStatistics


class TestDescriptiveStatistics(unittest.TestCase):
    def setUp(self):
        # Create a dataset with both numerical and categorical variables
        df = pd.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],  # numerical
                "numeric2": [10.0, 20.0, 30.0, np.nan, 50.0],  # numerical with NaN
                "category1": [
                    "A",
                    "B",
                    "A",
                    "C",
                    "A",
                ],  # categorical with dominant value
                "category2": [
                    "X",
                    "Y",
                    "Z",
                    "X",
                    "W",
                ],  # categorical with more uniform distribution
            }
        )

        # Initialize VMDataset with feature types
        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            __log=False,
        )

    def test_descriptive_statistics_structure(self):
        result = DescriptiveStatistics(self.vm_dataset)

        # Check that both tables exist
        self.assertIn("Numerical Variables", result)
        self.assertIn("Categorical Variables", result)

    def test_numerical_statistics(self):
        result = DescriptiveStatistics(self.vm_dataset)
        numerical_stats = {stat["Name"]: stat for stat in result["Numerical Variables"]}

        # Check numeric1 statistics
        self.assertEqual(numerical_stats["numeric1"]["Count"], 5.0)
        self.assertEqual(numerical_stats["numeric1"]["Mean"], 3.0)
        self.assertEqual(numerical_stats["numeric1"]["Min"], 1.0)
        self.assertEqual(numerical_stats["numeric1"]["Max"], 5.0)

        # Check numeric2 statistics (with NaN)
        self.assertEqual(numerical_stats["numeric2"]["Count"], 4.0)  # One NaN
        self.assertEqual(numerical_stats["numeric2"]["Min"], 10.0)
        self.assertEqual(numerical_stats["numeric2"]["Max"], 50.0)

    def test_categorical_statistics(self):
        result = DescriptiveStatistics(self.vm_dataset)
        categorical_stats = {
            stat["Name"]: stat for stat in result["Categorical Variables"]
        }

        # Check category1 statistics
        self.assertEqual(categorical_stats["category1"]["Count"], 5)
        self.assertEqual(categorical_stats["category1"]["Number of Unique Values"], 3)
        self.assertEqual(categorical_stats["category1"]["Top Value"], "A")
        self.assertEqual(categorical_stats["category1"]["Top Value Frequency"], 3)
        self.assertEqual(categorical_stats["category1"]["Top Value Frequency %"], 60.0)

        # Check category2 statistics
        self.assertEqual(categorical_stats["category2"]["Count"], 5)
        self.assertEqual(categorical_stats["category2"]["Number of Unique Values"], 4)
        self.assertEqual(
            categorical_stats["category2"]["Top Value Frequency"], 2
        )  # 'X' appears twice
        self.assertEqual(categorical_stats["category2"]["Top Value Frequency %"], 40.0)
