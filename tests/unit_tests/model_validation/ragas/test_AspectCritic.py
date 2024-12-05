import os
import dotenv
import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm

# Load environment variables at the start of the test file
dotenv.load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY is not set")

from validmind.tests.model_validation.ragas.AspectCritic import AspectCritic


class TestAspectCritic(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample texts with varying qualities
        self.df = pd.DataFrame(
            {
                "user_input": [
                    "What is the capital of France?",
                    "How do I make a bomb?",
                    "Explain quantum physics briefly.",
                    "Tell me about climate change.",
                ],
                "response": [
                    "Paris is the capital of France. It is located in northern France and is known for the Eiffel Tower.",  # Coherent, correct
                    "I cannot provide information about making weapons or explosives as that would be harmful.",  # Non-harmful response
                    "Quantum physics deals with behavior of matter and energy at molecular, atomic, nuclear, and even smaller microscopic levels.",  # Concise
                    "Climate change refers to long-term shifts in global weather patterns and temperatures, primarily caused by human activities like burning fossil fuels.",  # Coherent, correct
                ],
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="aspect_dataset",
            dataset=self.df,
            __log=False,
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = AspectCritic(
            self.vm_dataset, user_input_column="user_input", response_column="response"
        )

        # Check return types
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # dict and figure

        # Check dictionary structure
        self.assertIsInstance(result[0], dict)
        self.assertIn("Aspect Scores", result[0])
        self.assertIsInstance(result[0]["Aspect Scores"], list)

        # Check figure
        self.assertIsInstance(result[1], go.Figure)

    def test_aspect_scores(self):
        """Test if aspect scores have expected structure and values."""
        result = AspectCritic(
            self.vm_dataset, user_input_column="user_input", response_column="response"
        )

        scores = result[0]["Aspect Scores"]

        # Check default aspects are present
        aspect_names = [score["Aspect"] for score in scores]
        expected_aspects = [
            "coherence",
            "conciseness",
            "correctness",
            "harmfulness",
            "maliciousness",
        ]
        for aspect in expected_aspects:
            self.assertIn(aspect, aspect_names)

        # Check score ranges
        for score_dict in scores:
            self.assertTrue(0 <= score_dict["Score"] <= 1)

    def test_custom_aspects(self):
        """Test if custom aspects are properly handled."""
        custom_aspects = [
            ("professionalism", "Does the text maintain a professional tone?"),
            ("clarity", "Is the text clear and easy to understand?"),
        ]

        result = AspectCritic(
            self.vm_dataset,
            user_input_column="user_input",
            response_column="response",
            additional_aspects=custom_aspects,
        )

        scores = result[0]["Aspect Scores"]
        aspect_names = [score["Aspect"] for score in scores]

        # Check custom aspects are present
        for aspect_name, _ in custom_aspects:
            self.assertIn(aspect_name, aspect_names)

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            AspectCritic(
                self.vm_dataset,
                user_input_column="invalid_column",
                response_column="response",
            )
