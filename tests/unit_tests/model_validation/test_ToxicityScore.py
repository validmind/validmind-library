import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind.tests.model_validation.ToxicityScore import ToxicityScore


class TestToxicityScore(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample texts with varying toxicity levels
        reference_texts = [
            "I love this product!",  # Non-toxic
            "You are an idiot!",  # Toxic
            "The weather is nice today.",  # Non-toxic
            "I hate you, you are stupid.",  # Toxic
            "Have a great day!",  # Non-toxic
            "This is a normal sentence.",  # Non-toxic
            "You are worthless!",  # Toxic
        ]

        # Create predictions with varying toxicity levels
        predicted_texts = [
            "This product is amazing!",  # Non-toxic (matching sentiment)
            "You are a fool!",  # Toxic (matching sentiment)
            "It's a beautiful sunny day.",  # Non-toxic (matching sentiment)
            "I despise you, you're dumb.",  # Toxic (matching sentiment)
            "Enjoy your day!",  # Non-toxic (matching sentiment)
            "This is a regular statement.",  # Non-toxic (matching sentiment)
            "You are useless!",  # Toxic (matching sentiment)
        ]

        # Create DataFrame
        self.df = pd.DataFrame(
            {
                "text": reference_texts,
                "target": reference_texts,
                "predictions": predicted_texts,
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="toxicity_dataset",
            dataset=self.df,
            target_column="target",
            text_column="text",
            __log=False,
        )

        # Initialize ValidMind model
        self.vm_model = vm.init_model(
            input_id="toxicity_model",
            attributes={
                "architecture": "TextModel",
                "language": "Python",
            },
            __log=False,
        )

        # Link predictions
        self.vm_dataset.assign_predictions(
            self.vm_model, prediction_column="predictions"
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = ToxicityScore(self.vm_dataset, self.vm_model)

        # Check return types
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertEqual(len(result), 7)  # 1 DataFrame + 6 figures

        # Check all figures are Plotly figures
        for fig in result[1:]:
            self.assertIsInstance(fig, go.Figure)

    def test_results_dataframe(self):
        """Test if results DataFrame has expected structure."""
        result_df = ToxicityScore(self.vm_dataset, self.vm_model)[0]

        # Check columns
        expected_columns = [
            "Metric",
            "Mean Score",
            "Median Score",
            "Max Score",
            "Min Score",
            "Standard Deviation",
            "Count",
        ]
        self.assertListEqual(list(result_df.columns), expected_columns)

        # Check metrics
        expected_metrics = [
            "Input Text Toxicity",
            "True Text Toxicity",
            "Predicted Text Toxicity",
        ]
        self.assertListEqual(result_df["Metric"].tolist(), expected_metrics)

        # Check scores are between 0 and 1
        score_columns = ["Mean Score", "Median Score", "Max Score", "Min Score"]
        for col in score_columns:
            self.assertTrue(all(0 <= score <= 1 for score in result_df[col]))

    def test_identical_texts(self):
        """Test behavior with identical non-toxic texts."""
        # Create dataset with identical positive texts
        identical_texts = ["This is a very positive and friendly message."] * 5

        df_identical = pd.DataFrame(
            {
                "text": identical_texts,
                "target": identical_texts,
                "predictions": identical_texts,
            }
        )

        vm_dataset_identical = vm.init_dataset(
            input_id="identical_dataset",
            dataset=df_identical,
            target_column="target",
            text_column="text",
            __log=False,
        )

        # Link predictions
        vm_dataset_identical.assign_predictions(
            self.vm_model, prediction_column="predictions"
        )

        result_df = ToxicityScore(vm_dataset_identical, self.vm_model)[0]

        # All toxicity scores should be similar for identical texts
        scores = result_df["Mean Score"]
        self.assertTrue(all(abs(scores.iloc[0] - score) < 1e-5 for score in scores))

        # Standard deviation should be very small for identical texts
        self.assertTrue(all(std < 1e-5 for std in result_df["Standard Deviation"]))

    def test_toxicity_classification(self):
        import evaluate

        # Load toxicity model
        toxicity = evaluate.load("toxicity")

        # Test specific examples
        non_toxic_text = "I love this product!"
        toxic_text = "You are an idiot!"
        neutral_text = "The weather is nice today."

        # Get toxicity scores
        non_toxic_score = toxicity.compute(predictions=[non_toxic_text])["toxicity"][0]
        toxic_score = toxicity.compute(predictions=[toxic_text])["toxicity"][0]
        neutral_score = toxicity.compute(predictions=[neutral_text])["toxicity"][0]

        # Verify toxicity classifications
        self.assertTrue(
            toxic_score > non_toxic_score
        )  # Toxic text should have higher score
        self.assertTrue(
            neutral_score < toxic_score
        )  # Neutral text should be less toxic
