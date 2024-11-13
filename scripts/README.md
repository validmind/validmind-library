# Creating a new test

All ValidMind tests are in the `validmind/tests/` directory. Each file should be named using Camel Case and should have a single test class that matches the file name. For example, `MyNewTest.py` should have the test `class MyNewTest`. This class should inherit from `validmind.vm_models.Metric` or `validmind.vm_models.ThresholdTest` depending on the type of test you are creating. 

The tests are separated into subdirectories based on the category and type of test. For example, `validmind/tests/model_validation/sklearn` contains all of the model validation tests for sklearn-compatible models. There are two subdirectories in this folder: `metrics/` and `threshold_tests/` that contain the different types of tests. Any sub category can be used here and the `__init__.py` file will automatically pick up the tests.

Please see the notebook `listing-and-loading-tests.ipynb` for more information and examples and to learn about how the directory relates to the test's ID which is used across the ValidMind Platform.

To create a new test, you can use the create_new_test.py script to generate a metric or threshold test. This script will create a new test file in the appropriate directory and will also create a new test class in that file. It is registered as a custom Poetry script in the `pyproject.toml` and it can be used as follows:

```bash
generate-test --help  # see the usage instructions
generate-test  # interactively create a new test (will prompt for the test type and ID)
generate-test --test_type metric --test_id validmind.model_validation.sklearn.MyNewMetric  # create a new metric test for sklearn models
generate-test --test_type threshold_test --test_id validmind.data_validation.MyNewDataTest  # create a new threshold test for data validation
```
