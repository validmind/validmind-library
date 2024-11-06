# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd

from validmind.utils import format_dataframe

from .load import list_tests


def list_tags():
    """
    List unique tags from all test classes.
    """

    unique_tags = set()

    for test_id in list_tests():
        unique_tags.update(test.tags)

    return list(unique_tags)


def list_tasks_and_tags():
    """
    List all task types and their associated tags, with one row per task type and
    all tags for a task type in one row.

    Returns:
        pandas.DataFrame: A DataFrame with 'Task Type' and concatenated 'Tags'.
    """
    task_tags_dict = {}

    for test in list_tests():
        for task in test.tasks:
            task_tags_dict.setdefault(task, set()).update(test.tags)

    return format_dataframe(
        pd.DataFrame(
            [
                {"Task": task, "Tags": ", ".join(tags)}
                for task, tags in task_tags_dict.items()
            ]
        )
    )


def list_tasks():
    """
    List unique tasks from all test classes.
    """

    unique_tasks = set()

    for test in list_tests(__as_class=True):
        unique_tasks.update(test.tasks)

    return list(unique_tasks)
