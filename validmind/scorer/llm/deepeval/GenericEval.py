from typing import List

from validmind import tags, tasks
from validmind.errors import MissingDependencyError
from validmind.tests.decorator import scorer
from validmind.vm_models.dataset import VMDataset

try:
    from deepeval.metrics import GEval
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
@tags("llm", "GEval", "deepeval")
@tasks("llm")
def GenericEval(
    dataset: VMDataset,
    input_column: str = "input",
    actual_output_column: str = "actual_output",
    context_column: str = "context",
    metric_name: str = "Generic Evaluation",
    criteria: str = "Evaluate the response quality",
    evaluation_params: List[str] = None,
    threshold: float = 0.5,
):
    # Handle default evaluation_params
    if evaluation_params is None:
        evaluation_params = ["input", "actual_output", "context"]

    # Custom metric 1: Technical Accuracy
    geval_metric = GEval(
        name=metric_name,
        criteria=criteria,
        evaluation_params=evaluation_params,
        threshold=threshold,
    )

    results = []

    for _, row in dataset.df.iterrows():
        test_case = LLMTestCase(
            input=row[input_column],
            actual_output=row[actual_output_column],
            context=row[context_column],
        )
        geval_metric.measure(test_case)
        results.append({"score": geval_metric.score, "reason": geval_metric.reason})

    return results
