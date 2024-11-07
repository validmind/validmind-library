#' Register a Custom Test Function in ValidMind
#'
#' Registers an R function as a custom test within the ValidMind testing framework, allowing it to be used as a custom metric for model validation.
#'
#' @param func An R function to be registered as a custom test.
#' @param test_id A unique identifier for the test. If \code{NULL}, a default ID is generated based on the function name.
#' @param description A description of the test. If \code{NULL}, the function's \code{description} attribute is used. Defaults to "No description" if not available.
#' @param required_inputs A character vector specifying the required inputs for the test. If \code{NULL}, the function's formal argument names are used.
#'
#' @details
#' The provided R function is converted into a Python callable using \code{\link[reticulate]{r_to_py}}. 
#' A Python class is then defined, inheriting from ValidMind's \code{Metric} class, which wraps this callable.
#' This custom test is registered within ValidMind's test store and can be used in the framework for model validation purposes.
#'
#' @return The test store object containing the newly registered custom test.
#'
#' @examples
#' \dontrun{
#' # Define a custom test function in R
#' my_custom_metric <- function(predictions, targets) {
#'   # Custom metric logic
#'   mean(abs(predictions - targets))
#' }
#'
#' # Register the custom test
#' register_custom_test(
#'   func = my_custom_metric,
#'   test_id = "custom.mae",
#'   description = "Custom Mean Absolute Error",
#'   required_inputs = c("predictions", "targets")
#' )
#' }
#'
#' @seealso \code{\link[reticulate]{r_to_py}}, \code{\link[reticulate]{import_main}}, \code{\link[reticulate]{py_run_string}}
#'
#' @import reticulate
#' @export
register_custom_test <- function(func, test_id = NULL, description = NULL, required_inputs = NULL) {
    
    # Generate a default test_id if not provided
    if (is.null(test_id)) {
        test_id <- paste0("validmind.custom_metrics.", deparse(substitute(func)))
    }
    
    # Use the provided description or fall back on the function's 'description' attribute
    if (is.null(description)) {
        description <- attr(func, "description")
        if (is.null(description)) {
            description <- "No description"
        }
    }
    
    # Use the provided required inputs or the formal argument names of the function
    if (is.null(required_inputs)) {
        required_inputs <- names(formals(func))
    }
    
    # Convert the R function to a Python callable
    py_func <- r_to_py(func)
    
    # Access the main Python namespace
    py_main <- import_main()
    py_main$py_func <- py_func
    
    # Define a Python class that calls the R function
    py_env <- py_run_string("
from validmind.vm_models import Metric
from validmind.tests._store import test_store

class CustomMetric(Metric):
    '''Description'''
    required_inputs = []
    default_params = {}

    def run(self):
        # Map inputs to the function's required inputs
        inputs_dict = {key: getattr(self.inputs, key) for key in self.required_inputs}
        return py_func(**inputs_dict)
    ", convert = TRUE)
    
    # Set the description and required inputs for the Python class
    CustomMetric <- py_env$CustomMetric
    CustomMetric$`__doc__` <- description
    CustomMetric$required_inputs <- required_inputs
    
    # Register the custom test in ValidMind's test store
    test_store <- import("validmind.tests._store")$test_store
    test_store$register_custom_test(test_id, CustomMetric)
    
    return(test_store)
}


#' Run a Custom Test using the ValidMind Framework
#'
#' This function runs a custom test using the ValidMind framework through Python's 
#' `validmind.vm_models`. It retrieves a custom test by `test_id`, executes it with the provided 
#' `inputs`, and optionally displays the result. The result is also logged.
#'
#' @param test_id A string representing the ID of the custom test to run.
#' @param inputs A list of inputs required for the custom test.
#' @param test_registry A reference to the test register object which provides the custom test class.
#' @param show A logical value. If TRUE, the result will be displayed. Defaults to FALSE.
#' 
#' @return An object representing the result of the test, with an additional log function.
#' 
#' @export
#' @importFrom reticulate import
#' 
#' @examples
#' \dontrun{
#' result <- run_custom_test("test123", my_inputs, test_registry, show = TRUE)
#' }
run_custom_test <- function(test_id, inputs, test_registry, show = FALSE) {
    
    # Import necessary classes from Python's validmind.vm_models
    TestContext <- reticulate::import("validmind.vm_models", as = "vm")$TestContext
    TestInput <- reticulate::import("validmind.vm_models", as = "vm")$TestInput

    # Prepare the inputs for the test
    my_inputs <- TestInput(inputs = inputs)
    
    # Retrieve the custom test class using the test_id
    TestClass <- test_registry$get_custom_test(test_id)
    
    # Initialize the test with the context and inputs
    test <- TestClass(
        test_id = test_id,
        context = TestContext(),
        inputs = my_inputs
    )
    
    # Run the test and create a figure from the result
    test$result <- test$run()
    
    # Optionally display the result
    if (show) {
        test$result$show()
    }
    
    # Import the build_result function from Python
    build_result <- import("validmind.tests.decorator", convert = TRUE)$`_build_result`
    
    # Log the test result
    result_wrapper <- build_result(
        results=test$result,
        test_id=test_id,
        inputs=list(),
        params=test$params,
        description=test$description()
    )

    # Return the test result object
    return(result_wrapper)
}
