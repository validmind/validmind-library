import unittest
import time
import os
import sys

from tabulate import tabulate

from validmind.logging import get_logger
from tests.unit_tests.utils import (
    print_uncovered_tests_summary,
    print_coverage_statistics,
)

# Limit OpenMP on Mac so it doesn't segfault:
#
# By limiting OpenMP to a single thread (OMP_NUM_THREADS=1), we:
#  • Prevent nested parallelism from creating too many threads.
#  • Simplify thread management, avoiding conflicts or resource contention.
#  • Allow other threading backends (e.g., Apple’s libdispatch or PyTorch's
#       thread pool) to manage parallelism more predictably.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

logger = get_logger(__name__)

KNOWN_FAILING_TESTS = [
    # Scores are not sensitive to toxic content. See test_Toxicity.py and test_ToxicityScore.py
    # for details.
    "unit_tests.data_validation.nlp.test_Toxicity",
    "unit_tests.model_validation.test_ToxicityScore",
    # RegardScore test fails due to a bug in the evaluate library's regard tool (v0.4.3).
    # The regard tool's internal processing has an issue with data type handling that causes
    # a ValueError when processing text inputs. This appears to be a bug in the regard tool
    # itself, not in our implementation.
    "unit_tests.model_validation.test_RegardScore",
]
SUCCESSFUL_TESTS = []
SKIPPED_TESTS = [
    # Skipping the tests that require an OpenAI API key
    "unit_tests.model_validation.ragas.test_AnswerCorrectness",
    "unit_tests.model_validation.ragas.test_AspectCritic",
    "unit_tests.model_validation.ragas.test_ContextEntityRecall",
    "unit_tests.model_validation.ragas.test_ContextPrecision",
    "unit_tests.model_validation.ragas.test_ContextPrecisionWithoutReference",
    "unit_tests.model_validation.ragas.test_ContextRecall",
    "unit_tests.model_validation.ragas.test_Faithfulness",
    "unit_tests.model_validation.ragas.test_NoiseSensitivity",
    "unit_tests.model_validation.ragas.test_ResponseRelevancy",
    "unit_tests.model_validation.ragas.test_SemanticSimilarity",
]
FAILED_TESTS = []


def print_test_summary():
    """
    Print a summary of the test results.
    """
    print(">>> Test Summary")
    print(">>> NOTE: Please review failing test cases directly in the output below.")

    test_summary = []
    # Add successful tests
    for test in SUCCESSFUL_TESTS:
        test_summary.append(
            [
                test["test_id"],
                test["num_tests"],
                "SUCCESS",
                test["execution_time"],
            ]
        )

    # Add known failing tests
    for test in KNOWN_FAILING_TESTS:
        test_summary.append([test, None, "KNOWN FAILURE", None])

    # Add failed tests
    for test in FAILED_TESTS:
        test_summary.append([test, None, "FAILED", None])

    # Add skipped tests
    for test in SKIPPED_TESTS:
        test_summary.append([test, None, "SKIPPED", None])

    print(
        tabulate(
            test_summary,
            headers=["Test ID", "Number of Tests", "Status", "Execution Time"],
            tablefmt="pretty",
        )
    )


def _find_test_files():
    """
    Find all test files in the tests/unit_tests directory.
    Returns a list of module paths for test files.
    """
    start_dir = "tests/unit_tests"
    test_files = []
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                # Convert file path to module path
                file_path = os.path.join(root, file)
                module_path = os.path.relpath(file_path, "tests").replace("/", ".")[
                    :-3
                ]  # remove .py
                test_files.append(module_path)
    return test_files


def run_test_files():
    """
    Run all test files individually and track results.
    Returns True if all tests passed, False otherwise.
    """
    loader = unittest.TestLoader()
    all_tests_passed = True
    test_files = _find_test_files()

    for test_file in sorted(test_files):
        # Skip known failing tests and skipped tests
        if test_file in KNOWN_FAILING_TESTS:
            logger.debug(f"Skipping known failing test: {test_file}")
            continue
        if test_file in SKIPPED_TESTS:
            logger.debug(f"Skipping test: {test_file}")
            continue

        logger.info(f"Running tests from: {test_file}")
        try:
            start_time = time.time()
            suite = loader.loadTestsFromName(f"tests.{test_file}")
            num_tests = suite.countTestCases()
            runner = unittest.TextTestRunner()
            result = runner.run(suite)
            execution_time = time.time() - start_time

            if result.wasSuccessful():
                SUCCESSFUL_TESTS.append(
                    {
                        "test_id": test_file,
                        "num_tests": num_tests,
                        "execution_time": f"{execution_time:.2f}s",
                    }
                )
            else:
                FAILED_TESTS.append(test_file)
                all_tests_passed = False
                logger.error(f"Tests failed in {test_file}")
        except Exception as e:
            FAILED_TESTS.append(test_file)
            all_tests_passed = False
            logger.error(f"Error running tests from {test_file}: {str(e)}")

    return all_tests_passed


all_tests_passed = run_test_files()

# Print summary of unit tests results
print_test_summary()

# Print summary of ValidMind tests not covered by unit tests
print_uncovered_tests_summary()

# Print coverage statistics
print_coverage_statistics()

# Exit with failure if any tests failed
if not all_tests_passed:
    raise Exception(
        f"Tests failed: {FAILED_TESTS}\n\n See output above for more details."
    )
