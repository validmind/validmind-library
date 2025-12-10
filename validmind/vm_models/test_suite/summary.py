# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass
from typing import List, Optional

import ipywidgets as widgets

from ...logging import get_logger
from ...utils import display, md_to_html
from ..html_renderer import StatefulHTMLRenderer
from ..result import ErrorResult
from .test_suite import TestSuiteSection, TestSuiteTest

logger = get_logger(__name__)


def id_to_name(id: str) -> str:
    """Convert an ID to a human-readable name."""
    # replace underscores, hyphens etc with spaces
    name = id.replace("_", " ").replace("-", " ").replace(".", " ")
    # capitalize each word
    name = " ".join([word.capitalize() for word in name.split(" ")])

    return name


@dataclass
class TestSuiteSectionSummary:
    """Represents a summary of a test suite section."""

    tests: List[TestSuiteTest]
    description: Optional[str] = None

    _widgets: List[widgets.Widget] = None

    def __post_init__(self):
        self._build_summary()

    def _add_description(self):
        """Add the section description to the summary."""
        if not self.description:
            return

        self._widgets.append(
            widgets.HTML(
                value=f'<div class="result">{md_to_html(self.description)}</div>'
            )
        )

    def _add_tests_summary(self):
        """Add the test results summary."""
        children = []
        titles = []

        for test in self.tests:
            children.append(test.result.to_widget())
            titles.append(
                f"❌ {test.result.name}: {test.name} ({test.test_id})"
                if isinstance(test.result, ErrorResult)
                else f"{test.result.name}: {test.name} ({test.test_id})"
            )

        self._widgets.append(widgets.Accordion(children=children, titles=titles))

    def _build_summary(self):
        """Build the complete summary."""
        self._widgets = []

        if self.description:
            self._add_description()

        self._add_tests_summary()

        self.summary = widgets.VBox(self._widgets)

    def display(self):
        """Display the summary."""
        display(self.summary)

    def to_html(self):
        """Generate HTML representation."""
        html_parts = [StatefulHTMLRenderer.get_base_css()]

        if self.description:
            html_parts.append(
                f'<div class="result">{md_to_html(self.description)}</div>'
            )

        # Create accordion content
        accordion_items = []
        accordion_titles = []

        for test in self.tests:
            if hasattr(test.result, "to_html"):
                accordion_items.append(test.result.to_html())
            else:
                # Fallback to widget rendering wrapped in HTML
                accordion_items.append(str(test.result.to_widget().value))

            title_prefix = "❌ " if isinstance(test.result, ErrorResult) else ""
            accordion_titles.append(
                f"{title_prefix}{test.result.name}: {test.name} ({test.test_id})"
            )

        if accordion_items:
            accordion_html = StatefulHTMLRenderer.render_accordion(
                accordion_items, accordion_titles
            )
            html_parts.append(accordion_html)

        return f'<div class="vm-test-suite-section">{"".join(html_parts)}</div>'


@dataclass
class TestSuiteSummary:
    """Represents a summary of a complete test suite."""

    title: str
    description: str
    sections: List[TestSuiteSection]
    show_link: bool = True

    _widgets: List[widgets.Widget] = None

    def __post_init__(self):
        """Initialize the summary after the dataclass is created."""
        self._build_summary()

    def _add_title(self):
        """Add the title to the summary."""
        title = f"""
        <h2>Test Suite Results: <i style="color: #DE257E">{self.title}</i></h2><hr>
        """.strip()

        self._widgets.append(widgets.HTML(value=title))

    def _add_results_link(self):
        """Add a link to documentation on ValidMind."""
        # avoid circular import
        from ...api_client import get_api_host, get_api_model

        ui_host = get_api_host().replace("/api/v1/tracking", "").replace("api", "app")
        link = f"{ui_host}model-inventory/{get_api_model()}"
        results_link = f"""
        <h3>
            Check out the updated documentation on
            <a href="{link}" target="_blank">ValidMind</a>.
        </h3>
        """.strip()

        self._widgets.append(widgets.HTML(value=results_link))

    def _add_description(self):
        """Add the test suite description to the summary."""
        self._widgets.append(
            widgets.HTML(
                value=f'<div class="result">{md_to_html(self.description)}</div>'
            )
        )

    def _add_sections_summary(self):
        """Append the section summary."""
        children = []
        titles = []

        for section in self.sections:
            if not section.tests:
                continue

            children.append(
                TestSuiteSectionSummary(
                    description=section.description,
                    tests=section.tests,
                ).summary
            )
            titles.append(id_to_name(section.section_id))

        self._widgets.append(widgets.Accordion(children=children, titles=titles))

    def _add_top_level_section_summary(self):
        """Add the top-level section summary."""
        self._widgets.append(
            TestSuiteSectionSummary(tests=self.sections[0].tests).summary
        )

    def _add_footer(self):
        """Add the footer."""
        footer = """
        <style>
            .result {
                margin: 10px 0;
                padding: 10px;
                background-color: #f1f1f1;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        </style>
        """.strip()

        self._widgets.append(widgets.HTML(value=footer))

    def _build_summary(self):
        """Build the complete summary."""
        self._widgets = []

        self._add_title()
        if self.show_link:
            self._add_results_link()
        self._add_description()
        if len(self.sections) == 1:
            self._add_top_level_section_summary()
        else:
            self._add_sections_summary()

        self.summary = widgets.VBox(self._widgets)

    def display(self):
        """Display the summary."""
        display(self.summary)

    def to_html(self):
        """Generate HTML representation of the complete test suite summary."""
        html_parts = [StatefulHTMLRenderer.get_base_css()]

        # Add title
        title_html = f"""
        <h2>Test Suite Results: <i style="color: #DE257E">{self.title}</i></h2><hr>
        """
        html_parts.append(title_html)

        # Add results link if needed
        if self.show_link:
            # avoid circular import
            from ...api_client import get_api_host, get_api_model

            ui_host = (
                get_api_host().replace("/api/v1/tracking", "").replace("api", "app")
            )
            link = f"{ui_host}model-inventory/{get_api_model()}"
            results_link_html = f"""
            <h3>
                Check out the updated documentation on
                <a href="{link}" target="_blank">ValidMind</a>.
            </h3>
            """
            html_parts.append(results_link_html)

        # Add description
        html_parts.append(f'<div class="result">{md_to_html(self.description)}</div>')

        # Add sections
        if len(self.sections) == 1:
            # Single section - render tests directly
            section_summary = TestSuiteSectionSummary(tests=self.sections[0].tests)
            html_parts.append(section_summary.to_html())
        else:
            # Multiple sections - create accordion
            section_items = []
            section_titles = []

            for section in self.sections:
                if not section.tests:
                    continue

                section_summary = TestSuiteSectionSummary(
                    description=section.description,
                    tests=section.tests,
                )
                section_items.append(section_summary.to_html())
                section_titles.append(id_to_name(section.section_id))

            if section_items:
                sections_accordion = StatefulHTMLRenderer.render_accordion(
                    section_items, section_titles
                )
                html_parts.append(sections_accordion)

        return f'<div class="vm-test-suite-summary">{"".join(html_parts)}</div>'
