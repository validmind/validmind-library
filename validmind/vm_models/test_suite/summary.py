# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass
from typing import List, Optional

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

    def display(self):
        """Display the summary."""
        display(self.to_html())

    def to_html(self):
        """Generate HTML representation."""
        html_parts = [StatefulHTMLRenderer.get_base_css()]

        if self.description:
            html_parts.append(
                f'<div class="result">{md_to_html(self.description)}</div>'
            )

        accordion_items = []
        accordion_titles = []

        for test in self.tests:
            if hasattr(test.result, "to_html"):
                accordion_items.append(test.result.to_html())
            else:
                # Fallback: create a simple HTML representation
                accordion_items.append(
                    f'<div class="vm-result"><p>Result: {test.result.name}</p></div>'
                )

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

    def display(self):
        """Display the summary."""
        display(self.to_html())

    def to_html(self):
        """Generate HTML representation of the complete test suite summary."""
        html_parts = [StatefulHTMLRenderer.get_base_css()]

        title_html = f"""
        <h2>Test Suite Results: <i style="color: #DE257E">{self.title}</i></h2><hr>
        """
        html_parts.append(title_html)

        if self.show_link:
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

        html_parts.append(f'<div class="result">{md_to_html(self.description)}</div>')

        if len(self.sections) == 1:
            section_summary = TestSuiteSectionSummary(tests=self.sections[0].tests)
            html_parts.append(section_summary.to_html())
        else:
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
