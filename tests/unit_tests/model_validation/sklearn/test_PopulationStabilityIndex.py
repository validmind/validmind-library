import unittest

import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import validmind as vm
from validmind.tests.model_validation.sklearn.PopulationStabilityIndex import (
    PopulationStabilityIndex,
)


def _build(n_classes=3, drop_class=None, drop_class_initial=None, drop_class_new=None):
    """Fit a model on ``n_classes`` and build initial/new VM datasets.

    ``drop_class`` removes that class from both dataset slices while leaving the
    model fit on the full class set -- the missing-class scenario. Pass
    ``drop_class_initial``/``drop_class_new`` instead to drop a class from only
    one of the two slices -- the class-membership-differs-between-slices scenario.
    """
    if drop_class is not None:
        drop_class_initial = drop_class
        drop_class_new = drop_class

    X, y = make_classification(
        n_samples=800,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=7,
    )
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.5, random_state=7
    )
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    vm_model = vm.init_model(input_id="psi_model", model=model, __log=False)

    # Two evaluation slices to compare (initial vs new).
    X_a, X_b, y_a, y_b = train_test_split(X_eval, y_eval, test_size=0.5, random_state=7)

    def _ds(input_id, X_, y_, drop_cls):
        df = pd.DataFrame(X_, columns=[f"f{i}" for i in range(5)])
        df["target"] = y_
        if drop_cls is not None:
            df = df[df["target"] != drop_cls].reset_index(drop=True)
        ds = vm.init_dataset(
            input_id=input_id, dataset=df, target_column="target", __log=False
        )
        ds.assign_predictions(vm_model)
        return ds

    return (
        vm_model,
        _ds("psi_initial", X_a, y_a, drop_class_initial),
        _ds("psi_new", X_b, y_b, drop_class_new),
    )


class TestPopulationStabilityIndexMulticlass(unittest.TestCase):
    def test_one_vs_rest_tables_and_traces(self):
        model, ds_initial, ds_new = _build(n_classes=3)
        tables, fig, raw = PopulationStabilityIndex([ds_initial, ds_new], model)

        self.assertIsInstance(tables, dict)
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(raw, vm.RawData)

        # One table and one raw entry per class; 3 traces per class subplot.
        self.assertEqual(len(tables), 3)
        self.assertEqual(set(raw.psi_raw), {"0", "1", "2"})
        self.assertEqual(len(fig.data), 9)


class TestPopulationStabilityIndexMissingClass(unittest.TestCase):
    def test_missing_class_computes_present_tables(self):
        # Model trained on 4 classes; both slices omit class 3. Previously the
        # shape guard skipped; alignment on classes_ makes it compute now.
        model, ds_initial, ds_new = _build(n_classes=4, drop_class=3)
        tables, fig, raw = PopulationStabilityIndex([ds_initial, ds_new], model)

        # Only the 3 present classes get tables/raw entries; absent class 3 is gone.
        self.assertEqual(len(tables), 3)
        self.assertEqual(set(raw.psi_raw), {"0", "1", "2"})
        self.assertNotIn("3", set(raw.psi_raw))
        self.assertEqual(len(fig.data), 9)


class TestPopulationStabilityIndexClassMembershipDiffers(unittest.TestCase):
    def test_new_dataset_missing_class_uses_initial_present_set(self):
        # Model trained on 4 classes; initial slice has all 4, new slice is
        # missing class 3. The present-class set is derived from the initial
        # dataset only (the new dataset only gets a column-width check, since PSI
        # compares score distributions rather than positives), so this should
        # compute all 4 per-class tables without skipping.
        model, ds_initial, ds_new = _build(n_classes=4, drop_class_new=3)
        tables, fig, raw = PopulationStabilityIndex([ds_initial, ds_new], model)

        self.assertEqual(len(tables), 4)
        self.assertEqual(set(raw.psi_raw), {"0", "1", "2", "3"})
        self.assertEqual(len(fig.data), 12)


if __name__ == "__main__":
    unittest.main()
