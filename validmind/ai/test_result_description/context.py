# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import multiprocessing

MIN_IMAGES_FOR_PARALLEL = 4
MAX_WORKERS = multiprocessing.cpu_count()


def parallel_downsample_images(base64_strings):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from test_result_description.image_processing import (
        downsample_image,  # type: ignore
    )

    num_images = len(base64_strings)

    if num_images < MIN_IMAGES_FOR_PARALLEL:
        return [downsample_image(img) for img in base64_strings]

    num_workers = min(num_images, MAX_WORKERS)

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(downsample_image, base64_strings)

    sys.path.pop(0)

    return results


class Context:
    def __init__(self, mode="local"):
        pass

    def load(self, input_data):
        # this task can accept a dict or a test result object from the ValidMind Library
        if isinstance(input_data, dict):
            return input_data

        # we are likely running outside of the ValidMind Library and need to convert
        # the test result object to a dictionary
        test_result = input_data

        try:
            from markdownify import markdownify as md
        except ImportError as e:
            raise ImportError(
                "Failed to import markdownify. Please install the package to use this task."
            ) from e

        input_data = {
            "test_name": test_result.result_id.split(".")[-1],
            "test_description": md(test_result.result_metadata[0]["text"]),
        }

        if hasattr(test_result, "metric") and test_result.metric.summary is not None:
            input_data["summary"] = test_result.metric.summary.serialize()
        elif (
            hasattr(test_result, "test_results")
            and test_result.test_results.summary is not None
        ):
            input_data["summary"] = test_result.test_results.summary.serialize()

        if test_result.figures:
            input_data["figures"] = parallel_downsample_images(
                [figure._get_b64_url() for figure in test_result.figures]
            )

        return input_data
