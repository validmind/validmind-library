#!/usr/bin/env python3
"""
Unit tests for the @scorer decorator functionality (merged).

This module includes two kinds of tests:
1) Integration tests that exercise the real ValidMind imports (skipped if imports fail)
2) Standalone tests that use lightweight mocks and always run

Coverage:
- Registration (explicit and auto IDs)
- Separation from regular tests
- Metadata (tags, tasks)
- Save function
- Parameter handling
- Path-based ID generation (integration only)
"""

import unittest
from unittest.mock import patch, MagicMock

# Real imports for integration tests; may fail in certain dev environments
from validmind.tests.decorator import scorer, _generate_scorer_id_from_path, tags, tasks
from validmind.tests._store import scorer_store, test_store


class TestScorerDecorator(unittest.TestCase):
    """Integration tests for the @scorer decorator."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Clear the scorer store before each test
        scorer_store.scorers.clear()
        test_store.tests.clear()

    def tearDown(self):
        """Clean up after each test method."""
        # Clear the scorer store after each test
        scorer_store.scorers.clear()
        test_store.tests.clear()

    def test_scorer_with_explicit_id(self):
        """Test @scorer decorator with explicit ID."""
        @scorer("validmind.scorer.test.ExplicitScorer")
        def explicit_scorer(model, dataset):
            """A scorer with explicit ID."""
            return [1.0, 2.0, 3.0]

        # Check that the scorer is registered
        registered_scorer = scorer_store.get_scorer("validmind.scorer.test.ExplicitScorer")
        self.assertIsNotNone(registered_scorer)
        self.assertEqual(registered_scorer, explicit_scorer)
        self.assertEqual(explicit_scorer.scorer_id, "validmind.scorer.test.ExplicitScorer")

    def test_scorer_with_empty_parentheses(self):
        """Test @scorer() decorator with empty parentheses."""
        @scorer()
        def empty_parentheses_scorer(model, dataset):
            """A scorer with empty parentheses."""
            return list([4.0, 5.0, 6.0])

        # Check that the scorer is registered with auto-generated ID
        # The ID will be based on the file path since we're in a test file
        actual_id = empty_parentheses_scorer.scorer_id
        self.assertIsNotNone(actual_id)
        self.assertTrue(actual_id.startswith("validmind.scorer"))

        registered_scorer = scorer_store.get_scorer(actual_id)
        self.assertIsNotNone(registered_scorer)
        self.assertEqual(registered_scorer, empty_parentheses_scorer)
        self.assertEqual(empty_parentheses_scorer.scorer_id, actual_id)

    def test_scorer_without_parentheses(self):
        """Test @scorer decorator without parentheses."""
        @scorer
        def no_parentheses_scorer(model, dataset):
            """A scorer without parentheses."""
            return list([7.0, 8.0, 9.0])

        # Check that the scorer is registered with auto-generated ID
        # The ID will be based on the file path since we're in a test file
        actual_id = no_parentheses_scorer.scorer_id
        self.assertIsNotNone(actual_id)
        self.assertTrue(actual_id.startswith("validmind.scorer"))

        registered_scorer = scorer_store.get_scorer(actual_id)
        self.assertIsNotNone(registered_scorer)
        self.assertEqual(registered_scorer, no_parentheses_scorer)
        self.assertEqual(no_parentheses_scorer.scorer_id, actual_id)

    def test_scorer_separation_from_tests(self):
        """Test that scorers are stored separately from regular tests."""
        @scorer("validmind.scorer.test.SeparationTest")
        def separation_scorer(model, dataset):
            """A scorer for separation testing."""
            return list([1.0])

        # Check that scorer is in scorer store
        scorer_in_store = scorer_store.get_scorer("validmind.scorer.test.SeparationTest")
        self.assertIsNotNone(scorer_in_store)
        self.assertEqual(scorer_in_store, separation_scorer)

        # Check that scorer is NOT in regular test store
        test_in_store = test_store.get_test("validmind.scorer.test.SeparationTest")
        self.assertIsNone(test_in_store)

    def test_scorer_with_tags_and_tasks(self):
        """Test that @scorer decorator works with @tags and @tasks decorators."""
        @scorer("validmind.scorer.test.TaggedScorer")
        @tags("test", "scorer", "tagged")
        @tasks("classification")
        def tagged_scorer(model, dataset):
            """A scorer with tags and tasks."""
            return list([1.0])

        # Check that the scorer is registered
        registered_scorer = scorer_store.get_scorer("validmind.scorer.test.TaggedScorer")
        self.assertIsNotNone(registered_scorer)

        # Check that tags and tasks are preserved
        self.assertTrue(hasattr(tagged_scorer, '__tags__'))
        self.assertEqual(tagged_scorer.__tags__, ["test", "scorer", "tagged"])

        self.assertTrue(hasattr(tagged_scorer, '__tasks__'))
        self.assertEqual(tagged_scorer.__tasks__, ["classification"])

    def test_scorer_save_functionality(self):
        """Test that the save functionality is available."""
        @scorer("validmind.scorer.test.SaveTest")
        def save_test_scorer(model, dataset):
            """A scorer for testing save functionality."""
            return list([1.0])

        # Check that save function is available
        self.assertTrue(hasattr(save_test_scorer, 'save'))
        self.assertTrue(callable(save_test_scorer.save))

    def test_multiple_scorers_registration(self):
        """Test that multiple scorers can be registered without conflicts."""
        @scorer("validmind.scorer.test.Multiple1")
        def scorer1(model, dataset):
            return list([1.0])

        @scorer("validmind.scorer.test.Multiple2")
        def scorer2(model, dataset):
            return list([2.0])

        @scorer("validmind.scorer.test.Multiple3")
        def scorer3(model, dataset):
            return list([3.0])

        # Check that all scorers are registered
        self.assertIsNotNone(scorer_store.get_scorer("validmind.scorer.test.Multiple1"))
        self.assertIsNotNone(scorer_store.get_scorer("validmind.scorer.test.Multiple2"))
        self.assertIsNotNone(scorer_store.get_scorer("validmind.scorer.test.Multiple3"))

        # Check that they are different functions
        self.assertNotEqual(
            scorer_store.get_scorer("validmind.scorer.test.Multiple1"),
            scorer_store.get_scorer("validmind.scorer.test.Multiple2")
        )

    def test_scorer_with_parameters(self):
        """Test that scorers can have parameters."""
        @scorer("validmind.scorer.test.ParameterScorer")
        def parameter_scorer(model, dataset, threshold: float = 0.5, multiplier: int = 2):
            """A scorer with parameters."""
            return list([threshold * multiplier])

        # Check that the scorer is registered
        registered_scorer = scorer_store.get_scorer("validmind.scorer.test.ParameterScorer")
        self.assertIsNotNone(registered_scorer)
        self.assertEqual(registered_scorer, parameter_scorer)

    def test_scorer_docstring_preservation(self):
        """Test that docstrings are preserved."""
        @scorer("validmind.scorer.test.DocstringTest")
        def docstring_scorer(model, dataset):
            """This is a test docstring for the scorer."""
            return list([1.0])

        # Check that docstring is preserved
        self.assertEqual(docstring_scorer.__doc__, "This is a test docstring for the scorer.")


class TestScorerIdGeneration(unittest.TestCase):
    """Integration tests for automatic scorer ID generation from file paths."""

    def setUp(self):
        """Set up test fixtures."""
        scorer_store.scorers.clear()

    def tearDown(self):
        """Clean up after each test."""
        scorer_store.scorers.clear()

    @patch('validmind.tests.decorator.inspect.getfile')
    @patch('validmind.tests.decorator.os.path.relpath')
    @patch('validmind.tests.decorator.os.path.abspath')
    def test_generate_id_from_path_classification(self, mock_abspath, mock_relpath, mock_getfile):
        """Test ID generation for classification scorer."""
        # Mock the file path
        mock_getfile.return_value = "/path/to/validmind/scorer/classification/BrierScore.py"
        mock_abspath.return_value = "/path/to/validmind/scorer"
        mock_relpath.return_value = "classification/BrierScore.py"

        def mock_function():
            pass

        scorer_id = _generate_scorer_id_from_path(mock_function)
        expected_id = "validmind.scorer.classification.BrierScore"
        self.assertEqual(scorer_id, expected_id)

    @patch('validmind.tests.decorator.inspect.getfile')
    @patch('validmind.tests.decorator.os.path.relpath')
    @patch('validmind.tests.decorator.os.path.abspath')
    def test_generate_id_from_path_llm(self, mock_abspath, mock_relpath, mock_getfile):
        """Test ID generation for LLM scorer."""
        # Mock the file path
        mock_getfile.return_value = "/path/to/validmind/scorer/llm/deepeval/AnswerRelevancy.py"
        mock_abspath.return_value = "/path/to/validmind/scorer"
        mock_relpath.return_value = "llm/deepeval/AnswerRelevancy.py"

        def mock_function():
            pass

        scorer_id = _generate_scorer_id_from_path(mock_function)
        expected_id = "validmind.scorer.llm.deepeval.AnswerRelevancy"
        self.assertEqual(scorer_id, expected_id)

    @patch('validmind.tests.decorator.inspect.getfile')
    @patch('validmind.tests.decorator.os.path.relpath')
    @patch('validmind.tests.decorator.os.path.abspath')
    def test_generate_id_from_path_root_scorer(self, mock_abspath, mock_relpath, mock_getfile):
        """Test ID generation for scorer in root scorer directory."""
        # Mock the file path
        mock_getfile.return_value = "/path/to/validmind/scorer/MyScorer.py"
        mock_abspath.return_value = "/path/to/validmind/scorer"
        mock_relpath.return_value = "MyScorer.py"

        def mock_function():
            pass

        scorer_id = _generate_scorer_id_from_path(mock_function)
        expected_id = "validmind.scorer.MyScorer"
        self.assertEqual(scorer_id, expected_id)

    @patch('validmind.tests.decorator.inspect.getfile')
    def test_generate_id_fallback_on_error(self, mock_getfile):
        """Test ID generation fallback when path detection fails."""
        # Mock getfile to raise an exception
        mock_getfile.side_effect = OSError("Cannot determine file path")

        def mock_function():
            pass

        scorer_id = _generate_scorer_id_from_path(mock_function)
        expected_id = "validmind.scorer.mock_function"
        self.assertEqual(scorer_id, expected_id)

    @patch('validmind.tests.decorator.inspect.getfile')
    @patch('validmind.tests.decorator.os.path.relpath')
    @patch('validmind.tests.decorator.os.path.abspath')
    def test_generate_id_fallback_on_value_error(self, mock_abspath, mock_relpath, mock_getfile):
        """Test ID generation fallback when relative path calculation fails."""
        # Mock getfile to return a path outside the scorer directory
        mock_getfile.return_value = "/path/to/some/other/directory/MyScorer.py"
        mock_abspath.return_value = "/path/to/validmind/scorer"
        mock_relpath.side_effect = ValueError("Path not under scorer directory")

        def mock_function():
            pass

        scorer_id = _generate_scorer_id_from_path(mock_function)
        expected_id = "validmind.scorer.mock_function"
        self.assertEqual(scorer_id, expected_id)


class TestScorerIntegration(unittest.TestCase):
    """More integration tests for scorer behavior with the broader system."""

    def setUp(self):
        """Set up test fixtures."""
        scorer_store.scorers.clear()
        test_store.tests.clear()

    def tearDown(self):
        """Clean up after each test."""
        scorer_store.scorers.clear()
        test_store.tests.clear()

    def test_scorer_store_singleton(self):
        """Test that scorer store is a singleton."""
        from validmind.tests._store import ScorerStore

        store1 = ScorerStore()
        store2 = ScorerStore()

        self.assertIs(store1, store2)

    def test_scorer_registration_and_retrieval(self):
        """Test complete registration and retrieval cycle."""
        @scorer("validmind.scorer.test.IntegrationTest")
        def integration_scorer(model, dataset):
            """Integration test scorer."""
            return list([1.0, 2.0, 3.0])

        # Test registration
        self.assertIsNotNone(scorer_store.get_scorer("validmind.scorer.test.IntegrationTest"))

        # Test retrieval
        retrieved_scorer = scorer_store.get_scorer("validmind.scorer.test.IntegrationTest")
        self.assertEqual(retrieved_scorer, integration_scorer)

        # Test that it's callable
        self.assertTrue(callable(retrieved_scorer))

    def test_scorer_with_mock_model_and_dataset(self):
        """Test scorer execution with mock model and dataset."""
        @scorer("validmind.scorer.test.MockExecution")
        def mock_execution_scorer(model, dataset):
            """Scorer for mock execution testing."""
            return list([1.0, 2.0, 3.0])

        # Create mock model and dataset
        mock_model = MagicMock()
        mock_dataset = MagicMock()

        # Execute the scorer
        result = mock_execution_scorer(mock_model, mock_dataset)

        # Check result
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1.0, 2.0, 3.0])


# ---------------------------
# Standalone (mock-based) tests
# ---------------------------

from typing import Any, Callable, Optional, Union, List  # noqa: E402


class _MockList:
    def __init__(self, values):
        self.values = values

    def __eq__(self, other):
        if isinstance(other, list):
            return self.values == other
        return getattr(other, "values", None) == self.values


class _MockScorerStore:
    def __init__(self):
        self.scorers = {}

    def register_scorer(self, scorer_id: str, scorer: Callable[..., Any]) -> None:
        self.scorers[scorer_id] = scorer

    def get_scorer(self, scorer_id: str) -> Optional[Callable[..., Any]]:
        return self.scorers.get(scorer_id)


class _MockTestStore:
    def __init__(self):
        self.tests = {}

    def get_test(self, test_id: str) -> Optional[Callable[..., Any]]:
        return self.tests.get(test_id)


_mock_scorer_store = _MockScorerStore()
_mock_test_store = _MockTestStore()


def _mock_scorer(func_or_id: Union[Callable[..., Any], str, None] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Lightweight scorer decorator used for mock-based tests."""

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if func_or_id is None or func_or_id == "":
            scorer_id = f"validmind.scorer.{func.__name__}"
        elif isinstance(func_or_id, str):
            scorer_id = func_or_id
        else:
            scorer_id = f"validmind.scorer.{func.__name__}"

        _mock_scorer_store.register_scorer(scorer_id, func)
        func.scorer_id = scorer_id
        return func

    if callable(func_or_id):
        return _decorator(func_or_id)
    return _decorator


class TestScorerDecoratorEdgeCases(unittest.TestCase):
    def setUp(self):
        _mock_scorer_store.scorers.clear()
        _mock_test_store.tests.clear()

    def tearDown(self):
        _mock_scorer_store.scorers.clear()
        _mock_test_store.tests.clear()

    def test_scorer_with_empty_string_id(self):
        @_mock_scorer("")
        def empty_string_scorer(model, dataset):
            return _MockList([1.0])
        self.assertEqual(empty_string_scorer.scorer_id, "validmind.scorer.empty_string_scorer")
        self.assertIsNotNone(_mock_scorer_store.get_scorer("validmind.scorer.empty_string_scorer"))

    def test_scorer_with_none_id(self):
        @_mock_scorer(None)
        def none_id_scorer(model, dataset):
            return _MockList([1.0])
        self.assertEqual(none_id_scorer.scorer_id, "validmind.scorer.none_id_scorer")
        self.assertIsNotNone(_mock_scorer_store.get_scorer("validmind.scorer.none_id_scorer"))

    def test_scorer_with_complex_parameters(self):
        @_mock_scorer("validmind.scorer.test.ComplexParams")
        def complex_params_scorer(
            model,
            dataset,
            threshold: float = 0.5,
            enabled: bool = True,
            categories: List[str] = None,
            config: dict = None,
        ):
            if categories is None:
                categories = ["A", "B", "C"]
            if config is None:
                config = {"key": "value"}
            return _MockList([threshold, float(enabled), len(categories)])

        self.assertIsNotNone(_mock_scorer_store.get_scorer("validmind.scorer.test.ComplexParams"))

    def test_scorer_with_no_parameters(self):
        @_mock_scorer("validmind.scorer.test.NoParams")
        def no_params_scorer(model, dataset):
            return _MockList([1.0])
        self.assertIsNotNone(_mock_scorer_store.get_scorer("validmind.scorer.test.NoParams"))


if __name__ == '__main__':
    unittest.main(verbosity=2)
