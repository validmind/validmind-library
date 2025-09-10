# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from validmind.tests._store import test_provider_store
from validmind.tests.decorator import scorer
from validmind.tests.load import describe_test
from validmind.tests.run import run_test


def list_scorers(**kwargs):
    """List all scorers"""
    vm_provider = test_provider_store.get_test_provider("validmind")
    vm_scorers_provider = vm_provider.scorers_provider

    prefix = "validmind.scorer."

    return [
        f"{prefix}{test_id}" for test_id in vm_scorers_provider.list_tests(**kwargs)
    ]


def describe_scorer(scorer_id: str, **kwargs):
    """Describe a scorer"""
    return describe_test(scorer_id, **kwargs)


def run_scorer(scorer_id: str, **kwargs):
    """Run a scorer"""
    from validmind.tests._store import scorer_store

    # First check if it's a custom scorer in the scorer store
    custom_scorer = scorer_store.get_scorer(scorer_id)
    if custom_scorer is not None:
        # Run the custom scorer directly
        from inspect import getdoc

        from validmind.tests.load import _inspect_signature
        from validmind.tests.run import _get_test_kwargs, build_test_result

        # Set inputs and params attributes on the scorer function (like load_test does)
        if not hasattr(custom_scorer, "inputs") or not hasattr(custom_scorer, "params"):
            custom_scorer.inputs, custom_scorer.params = _inspect_signature(
                custom_scorer
            )

        input_kwargs, param_kwargs = _get_test_kwargs(
            test_func=custom_scorer,
            inputs=kwargs.get("inputs", {}),
            params=kwargs.get("params", {}),
        )

        raw_result = custom_scorer(**input_kwargs, **param_kwargs)

        return build_test_result(
            outputs=raw_result,
            test_id=scorer_id,
            test_doc=getdoc(custom_scorer),
            inputs=input_kwargs,
            params=param_kwargs,
            title=kwargs.get("title"),
            test_func=custom_scorer,
        )

    # Fall back to the test system for built-in scorers
    return run_test(scorer_id, **kwargs)


__all__ = ["list_scorers", "describe_scorer", "run_scorer", "scorer"]
