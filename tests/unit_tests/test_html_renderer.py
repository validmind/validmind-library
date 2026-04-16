"""Tests for progress HTML rendering."""

from validmind.vm_models.html_renderer import StatefulHTMLRenderer


def test_render_progress_bar_includes_progress_css():
    result = StatefulHTMLRenderer.render_progress_bar(
        value=1,
        max_value=4,
        description="Generating text for monitoring_plan...",
        bar_id="progress-test",
    )

    assert ".vm-progress-description" in result
    assert "font-weight: bold;" in result
    assert "Generating text for monitoring_plan..." in result


def test_render_live_progress_bar_includes_progress_css():
    result = StatefulHTMLRenderer.render_live_progress_bar(
        max_value=4,
        description="Running test suite...",
        bar_id="progress-live-test",
    )

    assert ".vm-progress-description" in result
    assert "font-weight: bold;" in result
    assert "window.updateProgress_progress_live_test" in result
