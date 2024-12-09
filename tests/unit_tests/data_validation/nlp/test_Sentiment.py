import unittest
import pandas as pd
import matplotlib.pyplot as plt
import validmind as vm
from validmind.tests.data_validation.nlp.Sentiment import Sentiment


class TestSentiment(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with text of varying sentiment
        self.df = pd.DataFrame(
            {
                "text": [
                    "I love this product, it is amazing!",  # Very positive
                    "This is terrible, I hate it.",  # Very negative
                    "The sky is blue.",  # Neutral
                    "The product weighs 5 pounds.",  # Neutral/Objective
                    "I think this might be good.",  # Slightly positive
                    "This makes me so angry!",  # Negative
                    "What a wonderful day!",  # Positive
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

    def test_returns_matplotlib_figure(self):
        # Run the function
        result = Sentiment(self.vm_dataset)

        # Check if result is a matplotlib Figure
        self.assertIsInstance(result, plt.Figure)

        # Check if figure has an axes
        self.assertTrue(len(result.axes) > 0)

        # Check if axes has a title and labels
        ax = result.axes[0]
        self.assertIsNotNone(ax.get_title())
        self.assertIsNotNone(ax.get_xlabel())

    def test_sentiment_range(self):
        from nltk.sentiment import SentimentIntensityAnalyzer

        # Initialize VADER
        sia = SentimentIntensityAnalyzer()

        # Get sentiment scores for our test data
        scores = [sia.polarity_scores(text)["compound"] for text in self.df["text"]]

        # Check that scores are within valid range [-1, 1]
        self.assertTrue(all(-1 <= score <= 1 for score in scores))

        # Check that we have both positive and negative sentiments
        self.assertTrue(any(score > 0 for score in scores))
        self.assertTrue(any(score < 0 for score in scores))

    def test_sentiment_distribution(self):
        from nltk.sentiment import SentimentIntensityAnalyzer

        # Initialize VADER
        sia = SentimentIntensityAnalyzer()

        # Test specific examples
        positive_text = "I love this product, it is amazing!"
        negative_text = "This is terrible, I hate it."
        neutral_text = "The sky is blue."

        # Get sentiment scores
        positive_score = sia.polarity_scores(positive_text)["compound"]
        negative_score = sia.polarity_scores(negative_text)["compound"]
        neutral_score = sia.polarity_scores(neutral_text)["compound"]

        # Verify sentiment classifications
        self.assertTrue(positive_score > 0.5)  # Strong positive
        self.assertTrue(negative_score < -0.5)  # Strong negative
        self.assertTrue(-0.2 <= neutral_score <= 0.2)  # Roughly neutral

    def test_raises_error_for_missing_text_column(self):
        # Check if ValueError is raised when text_column is not specified
        with self.assertRaises(ValueError):
            Sentiment(self.invalid_vm_dataset)
