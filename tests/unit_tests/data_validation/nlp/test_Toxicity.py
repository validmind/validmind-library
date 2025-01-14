import unittest
import pandas as pd
import matplotlib.pyplot as plt
import validmind as vm
from validmind.tests.data_validation.nlp.Toxicity import Toxicity
from validmind import RawData


class TestToxicity(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with text of varying toxicity
        self.df = pd.DataFrame(
            {
                "text": [
                    "I love this product!",  # Non-toxic
                    "You are an idiot!",  # Toxic
                    "The weather is nice today.",  # Non-toxic
                    "I hate you, you are stupid.",  # Toxic
                    "Have a great day!",  # Non-toxic
                    "This is a normal sentence.",  # Non-toxic
                    "You are worthless!",  # Toxic
                ]
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            text_column="text",
            __log=False,
        )

        # Create an invalid dataset without text column
        self.invalid_df = pd.DataFrame({"numeric": [1, 2, 3, 4, 5]})

        self.invalid_vm_dataset = vm.init_dataset(
            input_id="invalid_dataset",
            dataset=self.invalid_df,
            __log=False,
        )

    def test_returns_matplotlib_figure_and_raw_data(self):
        # Run the function
        result = Toxicity(self.vm_dataset)

        # Check if result is a tuple with two elements
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Check if the first element is a matplotlib Figure
        self.assertIsInstance(result[0], plt.Figure)

        # Check if figure has an axes
        self.assertTrue(len(result[0].axes) > 0)

        # Check if axes has a title and labels
        ax = result[0].axes[0]
        self.assertIsNotNone(ax.get_title())
        self.assertIsNotNone(ax.get_xlabel())

        # Check if the second element is an instance of RawData
        self.assertIsInstance(result[1], RawData)

    def test_toxicity_range(self):
        import evaluate

        # Load toxicity model
        toxicity = evaluate.load("toxicity")

        # Get toxicity scores for our test data
        scores = toxicity.compute(predictions=list(self.df["text"]))["toxicity"]

        print(f"Scores: {scores}")

        # Check that scores are within valid range [0, 1]
        self.assertTrue(all(0 <= score <= 1 for score in scores))

        # Check that we have both toxic and non-toxic content
        self.assertTrue(any(score > 0.5 for score in scores))  # Some toxic content
        self.assertTrue(any(score < 0.5 for score in scores))  # Some non-toxic content

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

    def test_raises_error_for_missing_text_column(self):
        # Check if ValueError is raised when text_column is not specified
        with self.assertRaises(ValueError):
            Toxicity(self.invalid_vm_dataset)
