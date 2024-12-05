import pathlib

from typing import List
from validmind.tests.load import _list_test_ids
from tabulate import tabulate


def test_ids_not_covered() -> List[str]:
    """
    Returns a list of test ids that are not covered by unit tests.

    Compares ValidMind test IDs (e.g., 'validmind.data_validation.ChiSquaredFeaturesTable')
    against unit test files (e.g., 'unit_tests.data_validation.test_ChiSquaredFeaturesTable')
    to find ValidMind tests that don't have corresponding unit tests.
    """
    vm_test_ids = _list_test_ids()
    unit_test_files = get_unit_test_files()

    # Convert unit test IDs to ValidMind test ID format
    unit_test_ids = {
        f"validmind.{'.'.join(test_id.split('.')[1:-1])}.{test_id.split('.')[-1][5:]}"
        for test_id in unit_test_files
    }

    return [test_id for test_id in vm_test_ids if test_id not in unit_test_ids]


def get_unit_test_files() -> List[str]:
    """
    Returns a list of all unit test files under tests/unit_tests directory.
    Returns them in the format: unit_tests.model_validation.ragas.test_ResponseRelevancy

    Returns:
        List[str]: List of test file paths in dot notation format
    """
    unit_tests_dir = pathlib.Path(__file__).parent
    test_files = []

    for test_file in unit_tests_dir.rglob("test_*.py"):
        # Convert to relative path from unit_tests directory
        relative_path = test_file.relative_to(unit_tests_dir)
        # Convert the path to dot notation and remove .py extension
        test_id = (
            str(relative_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        )
        # Add unit_tests prefix
        test_id = f"unit_tests.{test_id}"
        test_files.append(test_id)

    return sorted(test_files)


def print_uncovered_tests_summary():
    """
    Print a summary table of ValidMind tests that are not covered by unit tests.
    """
    uncovered_tests = test_ids_not_covered()

    # Create a list of single-item lists for tabulate
    summary_data = [[test_id] for test_id in sorted(uncovered_tests)]

    print("\n>>> Tests Without Unit Test Coverage")
    print(
        tabulate(
            summary_data,
            headers=["Test ID"],
            tablefmt="pretty",
        )
    )


def print_coverage_statistics():
    """
    Print coverage statistics including total coverage and coverage by area.
    """
    vm_test_ids = _list_test_ids()
    uncovered_tests = test_ids_not_covered()

    # Calculate total coverage
    total_tests = len(vm_test_ids)
    covered_tests = total_tests - len(uncovered_tests)
    coverage_percentage = (covered_tests / total_tests * 100) if total_tests > 0 else 0

    # Group tests by area
    area_stats = {}
    for test_id in vm_test_ids:
        # Split the test ID and extract area (everything between 'validmind.' and the last component)
        parts = test_id.split(".")
        area = ".".join(parts[1:-1])

        if area not in area_stats:
            area_stats[area] = {"total": 0, "uncovered": 0}
        area_stats[area]["total"] += 1

    # Count uncovered tests by area
    for test_id in uncovered_tests:
        parts = test_id.split(".")
        area = ".".join(parts[1:-1])
        area_stats[area]["uncovered"] += 1

    # Calculate coverage percentage for each area
    summary_data = []
    for area, stats in sorted(area_stats.items()):
        covered = stats["total"] - stats["uncovered"]
        coverage_pct = (covered / stats["total"] * 100) if stats["total"] > 0 else 0
        summary_data.append(
            [area, stats["total"], covered, stats["uncovered"], f"{coverage_pct:.1f}%"]
        )

    # Add total row
    summary_data.append(
        [
            "TOTAL",
            total_tests,
            covered_tests,
            len(uncovered_tests),
            f"{coverage_percentage:.1f}%",
        ]
    )

    # Print summary table
    print("\n>>> Coverage Statistics by Area")
    print(
        tabulate(
            summary_data,
            headers=["Area", "Total Tests", "Covered", "Uncovered", "Coverage"],
            tablefmt="pretty",
        )
    )
