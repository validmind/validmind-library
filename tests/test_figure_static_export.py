import os
import unittest
from unittest import mock

import plotly.graph_objects as go

from validmind.vm_models import figure as figure_module
from validmind.vm_models.figure import Figure


def _make_figure():
    return Figure(key="k", figure=go.Figure(go.Scatter(x=[1], y=[1])), ref_id="r")


class TestPlotlyStaticExportFallback(unittest.TestCase):
    def setUp(self):
        figure_module._static_export_unavailable = False
        self.addCleanup(setattr, figure_module, "_static_export_unavailable", False)

    def test_falls_back_to_placeholder_and_stops_retrying(self):
        fig = _make_figure()
        with mock.patch.object(
            go.FigureWidget,
            "to_image",
            side_effect=RuntimeError("Kaleido requires Google Chrome"),
        ) as to_image:
            first = fig._get_png_bytes()
            second = _make_figure()._get_png_bytes()

        self.assertTrue(first.startswith(b"\x89PNG"))
        self.assertEqual(first, second)
        self.assertEqual(to_image.call_count, 1)
        self.assertFalse(fig.has_static_image)

    def test_still_uploads_plotly_json(self):
        with mock.patch.object(
            go.FigureWidget, "to_image", side_effect=RuntimeError("no chrome")
        ):
            files = _make_figure().serialize_files()
        self.assertEqual(set(files), {"image", "json_image"})

    def test_env_var_skips_kaleido(self):
        with mock.patch.dict(os.environ, {"VALIDMIND_PLOTLY_STATIC_EXPORT": "false"}):
            with mock.patch.object(go.FigureWidget, "to_image") as to_image:
                fig = _make_figure()
                png = fig._get_png_bytes()
                self.assertFalse(fig.has_static_image)
        to_image.assert_not_called()
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_working_export_is_used(self):
        with mock.patch.object(
            go.FigureWidget, "to_image", return_value=b"real-png"
        ):
            fig = _make_figure()
            self.assertEqual(fig._get_png_bytes(), b"real-png")
            self.assertTrue(fig.has_static_image)


if __name__ == "__main__":
    unittest.main()
