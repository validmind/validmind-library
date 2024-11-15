import unittest
import os
import time
from tabulate import tabulate
from validmind.logging import get_logger

logger = get_logger(__name__)

KNOWN_FAILING_TESTS = [
    # Scores are not sensitive to toxic content. See test_Toxicity.py for details. Below are the scores from the test: 
    # Non-toxic score: 0.0001393833226757124, Toxic score: 0.00013222619600128382, Neutral score: 0.00016061807400546968
    "unit_tests.data_validation.nlp.test_Toxicity",
]
SUCCESSFUL_TESTS = []
SKIPPED_TESTS = []
FAILED_TESTS = []


def print_test_summary():
    """
    Print a summary of the test results.
    """
    logger.info(">>> Test Summary")
    logger.info(
        ">>> NOTE: Please review failing test cases directly in the output below."
    )

    test_summary = []
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
        # Skip known failing tests
        if test_file in KNOWN_FAILING_TESTS:
            logger.info(f"Skipping known failing test: {test_file}")
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

# Print the summary at the end
print_test_summary()

# Exit with appropriate code
exit(0 if all_tests_passed else 1)
