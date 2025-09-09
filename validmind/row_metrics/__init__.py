# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from validmind.tests._store import test_provider_store
from validmind.tests.load import describe_test
from validmind.tests.run import run_test


def list_row_metrics(**kwargs):
    """List all metrics"""
    vm_provider = test_provider_store.get_test_provider("validmind")
    vm_metrics_provider = vm_provider.row_metrics_provider

    prefix = "validmind.row_metrics."

    return [
        f"{prefix}{test_id}" for test_id in vm_metrics_provider.list_tests(**kwargs)
    ]


def describe_row_metric(metric_id: str, **kwargs):
    """Describe a metric"""
    return describe_test(metric_id, **kwargs)


def run_row_metric(metric_id: str, **kwargs):
    """Run a metric"""
    return run_test(metric_id, **kwargs)


__all__ = ["list_row_metrics", "describe_row_metric", "run_row_metric"]
