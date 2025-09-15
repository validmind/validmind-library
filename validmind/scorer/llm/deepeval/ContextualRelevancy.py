from validmind import tags, tasks
from validmind.errors import MissingDependencyError
from validmind.tests.decorator import scorer
from validmind.vm_models.dataset import VMDataset

try:
    from deepeval.metrics import ContextualRelevancyMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for ContextualRelevancyMetric. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "ContextualRelevancy", "deepeval")
@tasks("llm")
def ContextualRelevancy(dataset: VMDataset, threshold: float = 0.5):
    """
    Evaluates RAG system performance using deepeval's built-in metrics.

    Args:
        dataset: VMDataset containing RAG test cases with input, actual_output,
                expected_output, context and retrieval_context
        params: Optional parameters for metric configuration

    Returns:
        Dictionary containing evaluation results from multiple deepeval metrics
    """
    # Initialize metrics
    context_relevancy = ContextualRelevancyMetric(threshold=threshold)

    results = []

    # Evaluate each test case
    for _, row in dataset.df.iterrows():
        test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"],
            context=[row["context"]],
            retrieval_context=[row["retrieval_context"]],
        )

        # Run metrics
        context_relevancy.measure(test_case)

        # Store results
        results.append(
            {"score": context_relevancy.score, "reason": context_relevancy.reason}
        )

    return results
