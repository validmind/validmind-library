import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import validmind as vm
from validmind import RawData
from validmind.tests.model_validation.sklearn.FeatureImportance import FeatureImportance


class TestFeatureImportance(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with known feature importance
        np.random.seed(42)
        n_samples = 100

        # Create features with different importance levels
        features = {
            "important_feature": np.random.normal(0, 1, n_samples)
            * 10,  # High importance
            "medium_feature": np.random.normal(0, 1, n_samples)
            * 5,  # Medium importance
            "weak_feature": np.random.normal(0, 1, n_samples),  # Low importance
            "noise_feature": np.random.normal(0, 0.1, n_samples),  # Noise
        }

        # Create target variable with known relationship to features
        target = (
            features["important_feature"] * 2
            + features["medium_feature"] * 1
            + features["weak_feature"] * 0.5
            + np.random.normal(0, 0.1, n_samples)
        )  # Add some noise

        # Combine features and target into a single DataFrame
        features["target"] = target
        self.df = pd.DataFrame(features)

        # Create and train a simple model
        X = self.df.drop("target", axis=1)
        y = self.df["target"]
        self.model = LinearRegression()
        self.model.fit(X, y)

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            target_column="target",
            __log=False,
        )

        # Wrap model in ValidMind model object
        self.vm_model = vm.init_model(
            input_id="model",
            model=self.model,
            __log=False,
        )

    def test_returns_dataframe_and_rawdata(self):
        # Run the function
        result_df, raw_data = FeatureImportance(self.vm_dataset, self.vm_model)

        # Check if result_df is a DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)

        # Check if DataFrame has expected columns
        expected_columns = ["Feature 1", "Feature 2", "Feature 3"]
        self.assertTrue(all(col in result_df.columns for col in expected_columns))

        # Check if raw_data is an instance of RawData
        self.assertIsInstance(raw_data, RawData)

    def test_feature_importance_ranking(self):
        # Run with all features
        result_df, _ = FeatureImportance(self.vm_dataset, self.vm_model, num_features=4)

        # Get feature names and scores
        features = []
        for i in range(1, 5):
            feature_info = result_df[f"Feature {i}"].iloc[0]
            feature_name = feature_info.split(";")[0].strip("[]")
            features.append(feature_name)

        # Check if important_feature is ranked first
        self.assertEqual(features[0], "important_feature")

        # Check if noise_feature is ranked last
        self.assertEqual(features[-1], "noise_feature")

    def test_num_features_parameter(self):
        # Test with different num_features values
        for num_features in [2, 3, 4]:
            result_df, _ = FeatureImportance(
                self.vm_dataset, self.vm_model, num_features=num_features
            )

            # Check number of columns matches num_features
            feature_columns = [
                col for col in result_df.columns if col.startswith("Feature")
            ]
            self.assertEqual(len(feature_columns), num_features)

    def test_feature_importance_scores(self):
        result_df, _ = FeatureImportance(self.vm_dataset, self.vm_model)

        # Get first feature score
        first_feature = result_df["Feature 1"].iloc[0]
        score = float(first_feature.split(";")[1].strip("[] "))

        # Check if score is positive (since we're using absolute importance)
        self.assertGreater(score, 0)

        # Check if score is finite
        self.assertTrue(np.isfinite(score))
