import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import validmind as vm
from validmind.tests.model_validation.ModelPredictionResiduals import ModelPredictionResiduals


class TestModelPredictionResiduals(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample time series data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create predictable pattern with some noise
        X = np.arange(100).reshape(-1, 1)
        y_true = 2 * X.ravel() + np.random.normal(0, 1, 100)  # Linear pattern with noise
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'feature': X.ravel(),
            'target': y_true
        }, index=dates)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y_true)
        
        # Initialize ValidMind dataset and model
        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=self.df,
            target_column='target',
            __log=False,
        )
        
        self.vm_model = vm.init_model(
            input_id="test_model",
            model=model,
            __log=False,
        )
        
        # Link predictions
        self.vm_dataset.assign_predictions(self.vm_model)

    def test_return_structure(self):
        """Test if function returns expected structure (DataFrame and two figures)."""
        result = ModelPredictionResiduals(self.vm_dataset, self.vm_model)
        
        # Should return a tuple of (DataFrame, Figure, Figure)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], go.Figure)
        self.assertIsInstance(result[2], go.Figure)

    def test_summary_dataframe_columns(self):
        """Test if summary DataFrame contains expected columns."""
        summary_df = ModelPredictionResiduals(self.vm_dataset, self.vm_model)[0]
        
        expected_columns = [
            'KS Statistic',
            'p-value',
            'KS Normality',
            'p-value Threshold'
        ]
        
        self.assertListEqual(list(summary_df.columns), expected_columns)

    def test_date_filtering(self):
        """Test if date filtering works correctly."""
        start_date = '2023-02-01'
        end_date = '2023-03-01'
        
        result_df = ModelPredictionResiduals(
            self.vm_dataset, 
            self.vm_model,
            start_date=start_date,
            end_date=end_date
        )[0]
        
        # Results should still contain all summary statistics
        self.assertIn('KS Statistic', result_df.columns)
        self.assertIn('p-value', result_df.columns)

    def test_p_value_threshold(self):
        """Test if p_value_threshold affects normality determination."""
        custom_threshold = 0.01
        summary_df = ModelPredictionResiduals(
            self.vm_dataset, 
            self.vm_model,
            p_value_threshold=custom_threshold
        )[0]
        
        self.assertEqual(summary_df['p-value Threshold'].iloc[0], custom_threshold)
