"""This is a test harness to run unit tests against the ValidMind tests"""

import os
import time
import unittest
from fnmatch import fnmatch

import matplotlib.pyplot as plt

from tabulate import tabulate
from tqdm import tqdm
from validmind.logging import get_logger
from validmind.tests import list_tests, load_test, run_test
from validmind.vm_models.test.result_wrapper import (
    MetricResultWrapper,
    ThresholdTestResultWrapper,
)

from run_test_utils import (
    setup_clustering_test_inputs,
    setup_embeddings_test_inputs,
    setup_summarization_test_inputs,
    setup_tabular_test_inputs,
    setup_time_series_test_inputs,
)


logger = get_logger(__name__)
# Override plt.show to do nothing
plt.show = lambda: None

# These tests are expected to fail and need to be fixed
KNOWN_FAILING_TESTS = [
    # Only statsmodels are supported for these metrics
    "validmind.model_validation.statsmodels.RegressionCoeffs",
    "validmind.model_validation.statsmodels.RegressionFeatureSignificance",
    # The number of observations is too small to use the Zivot-Andrews test
    "validmind.data_validation.ZivotAndrewsArch",
    # These tests can be turned into comparison tests:
    "validmind.model_validation.embeddings.CosineSimilarityComparison",
    "validmind.model_validation.embeddings.EuclideanDistanceComparison",
    # This is a base test class so it doesn't run on its own
    "validmind.model_validation.sklearn.ClusterPerformance",
    # ValueError: The `cluster_column` parameter must be provided
    "validmind.model_validation.embeddings.EmbeddingsVisualization2D",
]
SKIPPED_TESTS = []
SUCCESSFUL_TESTS = []

# Harcode some tests that require specific inputs instead of trying to
# guess from tags or tasks
CUSTOM_TEST_INPUT_ASSIGNMENTS = {
    "validmind.data_validation.DatasetDescription": "classification",
    "validmind.data_validation.DatasetSplit": "classification",
    "validmind.model_validation.ModelMetadata": "classification",
}

# Some tests require specific configurations. This is either expected and required
# or we need to fix these tests so they can run with sane defaults
#
# Here we assign config param keys to each test and then let the test runner know
# how to load the config for that test
TEST_TO_PARAMS_CONFIG = {
    # TODO: features_pairs should default to all input dataset pairs
    "validmind.model_validation.statsmodels.ScorecardHistogram": "score_column",
    # TODO: "ValueError: perplexity must be less than n_samples if using defaults"
    "validmind.model_validation.embeddings.TSNEComponentsPairwisePlots": "t_sne_config",
    "validmind.model_validation.sklearn.KMeansClustersOptimization": "kmeans_config",
    "validmind.model_validation.sklearn.HyperParametersTuning": "hyperparameter_tuning_config",
    "validmind.model_validation.embeddings.StabilityAnalysisKeyword": "stability_analysis_keyword_config",
}

# Global inputs and configurations for the tests
TEST_CONFIG = {}
TEST_INPUTS = {}


class TestRunTest(unittest.TestCase):
    pass


def create_unit_test_func(vm_test_id, vm_test_class):
    def unit_test_func(self):
        self.assertTrue(
            hasattr(vm_test_class, "required_inputs"),
            f"{vm_test_id} missing required_inputs",
        )
        self.assertTrue(
            len(vm_test_class.tasks) > 0,
            f"{vm_test_id} missing tasks in metadata",
        )
        self.assertTrue(
            len(vm_test_class.tags) > 0,
            f"{vm_test_id} missing tags in metadata",
        )

        required_inputs = sorted(vm_test_class.required_inputs)
        if required_inputs == ["datasets", "models"]:
            logger.debug(
                "Skipping test - multi-(dataset,model) tests are not supported at the moment %s",
                vm_test_id,
            )
            SKIPPED_TESTS.append(vm_test_id)
            return

        if "llm" in vm_test_class.tags and "embeddings" not in vm_test_class.tags:
            logger.debug(
                "--- Skipping test - LLM tests not supported yet %s",
                vm_test_id,
            )
            SKIPPED_TESTS.append(vm_test_id)
            return

        logger.debug(">>> Running test %s", vm_test_id)

        # Assume we'll load the classification (tabular) inputs in most cases
        custom_test_input_assignment = CUSTOM_TEST_INPUT_ASSIGNMENTS.get(vm_test_id)
        selected_test_inputs = None

        if custom_test_input_assignment:
            selected_test_inputs = custom_test_input_assignment
        elif "clustering" in vm_test_class.tasks:
            selected_test_inputs = "clustering"
        elif "embeddings" in vm_test_class.tags:
            selected_test_inputs = "embeddings"
        elif (
            "text_summarization" in vm_test_class.tasks or "nlp" in vm_test_class.tasks
        ):
            selected_test_inputs = "text_summarization"
        elif "time_series_data" in vm_test_class.tags:
            selected_test_inputs = "time_series"
        else:
            selected_test_inputs = "classification"

        inputs = TEST_INPUTS[selected_test_inputs]

        # Build the single test inputs according to the required inputs
        single_test_inputs = {}
        if required_inputs == ["dataset"]:
            single_test_inputs = inputs["single_dataset"]
        elif required_inputs == ["dataset", "model"]:
            single_test_inputs = inputs["model_and_dataset"]
        elif required_inputs == ["datasets"]:
            single_test_inputs = inputs["two_datasets"]
        elif required_inputs == ["datasets", "model"]:
            single_test_inputs = inputs["model_and_two_datasets"]
        elif required_inputs == ["models"]:
            single_test_inputs = inputs["two_models"]
        elif required_inputs == ["dataset", "models"]:
            single_test_inputs = inputs["dataset_and_two_models"]
        elif required_inputs == ["model"]:
            single_test_inputs = inputs["single_model"]

        test_kwargs = {
            "test_id": vm_test_id,
            "inputs": single_test_inputs,
            "__log": False,
            "show": False,
        }

        # Check if the test requires a specific configuration
        if vm_test_id in TEST_TO_PARAMS_CONFIG:
            key = TEST_TO_PARAMS_CONFIG.get(vm_test_id)
            if key in TEST_CONFIG:
                test_config = TEST_CONFIG.get(key)
                # Only set the config if it's not None
                if test_config:
                    test_kwargs["params"] = test_config
            else:
                logger.error(
                    "Skipping test %s - missing expected configuration for %s",
                    vm_test_id,
                    key,
                )
                SKIPPED_TESTS.append(vm_test_id)
                return

        start_time = time.time()
        results = run_test(**test_kwargs)
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)

        self.assertTrue(
            isinstance(results, (MetricResultWrapper, ThresholdTestResultWrapper)),
            f"Expected MetricResultWrapper or ThresholdTestResultWrapper, got {type(results)}",
        )
        self.assertEqual(
            results.result_id,
            vm_test_id,
            f"Expected result_id to be {vm_test_id}, got {results.result_id}",
        )
        if isinstance(results, MetricResultWrapper):
            self.assertTrue(
                results.metric is not None or results.figures is not None,
                f"A metric result needs to produce a metric result or a figure",
            )

        if isinstance(results, ThresholdTestResultWrapper):
            self.assertTrue(
                results.test_results is not None or results.figures is not None,
                f"A threshold test needs to produce a test result or a figure",
            )

        # Finally, the test worked so we can add it to the list of successful tests
        # and note the time it took to run
        SUCCESSFUL_TESTS.append(
            {
                "test_id": vm_test_id,
                "test_input_types": selected_test_inputs,
                "execution_time": execution_time,
            }
        )

    return unit_test_func


def create_test_summary_func():
    """
    Create a function that prints a summary of the test results.
    We do this dynamically so it runs after all the tests have run.
    """

    def test_summary(self):
        self.assertTrue(
            True,
            "Test results not found. Did any tests run?",
        )
        logger.info(">>> Test Summary")
        logger.info(
            ">>> NOTE: Please review failing test cases directly in the output below."
        )

        test_summary = []
        for test in SUCCESSFUL_TESTS:
            test_summary.append(
                [
                    test["test_id"],
                    test["test_input_types"],
                    "SUCCESS",
                    test["execution_time"],
                ]
            )

        for test in KNOWN_FAILING_TESTS:
            test_summary.append([test, None, "KNOWN FAILURE", None])

        for test in SKIPPED_TESTS:
            test_summary.append([test, None, "SKIPPED", None])

        print(
            tabulate(
                test_summary,
                headers=["Test ID", "Type of Test Inputs", "Status", "Execution Time"],
                tablefmt="pretty",
            )
        )

    return test_summary


def create_unit_test_funcs_from_vm_tests():
    setup_tabular_test_inputs(TEST_INPUTS, TEST_CONFIG)
    setup_summarization_test_inputs(TEST_INPUTS, TEST_CONFIG)
    setup_time_series_test_inputs(TEST_INPUTS, TEST_CONFIG)
    setup_embeddings_test_inputs(TEST_INPUTS, TEST_CONFIG)
    setup_clustering_test_inputs(TEST_INPUTS, TEST_CONFIG)

    custom_test_ids = os.environ.get("TEST_IDS")
    custom_test_ids = custom_test_ids.split(",") if custom_test_ids else None
    tests_to_run = list_tests(pretty=False) if not custom_test_ids else custom_test_ids

    # allow filtering tests by wildcard using fnmatch
    # e.g. only run tests that start with "validmind.data_validation"
    # TEST_PATTERN="validmind.data_validation*"
    test_pattern = os.environ.get("TEST_PATTERN")
    if test_pattern:
        tests_to_run = [
            test_id for test_id in tests_to_run if fnmatch(test_id, test_pattern)
        ]

    for vm_test_id in tqdm(sorted(tests_to_run)):
        # Only skip known failing tests if we're not running a custom set of tests
        if custom_test_ids is None and vm_test_id in KNOWN_FAILING_TESTS:
            logger.debug("Skipping known failing test %s", vm_test_id)
            continue

        # load the test class
        vm_test_class = load_test(vm_test_id)

        # create a unit test function for the test class
        unit_test_func = create_unit_test_func(vm_test_id, vm_test_class)
        unit_test_func_name = f'test_{vm_test_id.replace(".", "_")}'

        # add the unit test function to the unit test class
        setattr(TestRunTest, f"test_{unit_test_func_name}", unit_test_func)

    # create a test summary function. the zzz is to ensure it runs last
    test_summary_func = create_test_summary_func()
    setattr(TestRunTest, "test_zzz_summary", test_summary_func)


create_unit_test_funcs_from_vm_tests()


if __name__ == "__main__":
    unittest.main()
