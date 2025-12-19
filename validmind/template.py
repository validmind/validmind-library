# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import uuid
from typing import Any, Dict, List, Optional, Type

from .html_templates.content_blocks import (
    failed_content_block_html,
    non_test_content_block_html,
)
from .logging import get_logger
from .tests import LoadTestError, describe_test
from .utils import display, is_notebook, test_id_to_name
from .vm_models import TestSuite
from .vm_models.html_renderer import StatefulHTMLRenderer

logger = get_logger(__name__)

CONTENT_TYPE_MAP = {
    "test": "Test",
    "metric": "Metric",
    "unit_metric": "Unit Metric",
    "metadata_text": "Metadata Text",
    "dynamic": "Dynamic Content",
    "text": "Text",
    "risk_assessment": "Risk Assessment",
    "assessment_summary": "Assessment Summary",
    "guideline": "Guideline Assessment",
}


def _convert_sections_to_section_tree(
    sections: List[Dict[str, Any]],
    parent_id: str = "_root_",
    start_section_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    section_tree = []

    for section in sections:
        section_parent_id = section.get("parent_section", "_root_")

        if start_section_id:
            if section["id"] == start_section_id:
                child_sections = _convert_sections_to_section_tree(
                    sections, section["id"]
                )
                section_tree.append({**section, "sections": child_sections})

        elif section_parent_id == parent_id:
            child_sections = _convert_sections_to_section_tree(sections, section["id"])
            section_tree.append({**section, "sections": child_sections})

    if start_section_id and not section_tree:
        raise ValueError(f"Section {start_section_id} not found in template")
    # sort the section tree by the order of the sections in the template (if provided)
    # set the order to 9999 for the sections that do not have an order
    return sorted(section_tree, key=lambda x: x.get("order", 9999))


def _render_test_accordion(content: str, title: str) -> str:
    """Render a test block accordion with styling matching text blocks.

    Args:
        content: HTML content for the accordion item
        title: Title for the accordion header

    Returns:
        HTML string with accordion matching text block styling
    """
    accordion_id = f"accordion-{uuid.uuid4().hex[:8]}"
    item_id = f"{accordion_id}-item-0"

    return f"""
        <div class="vm-accordion" id="{accordion_id}">
            <div class="vm-accordion-item">
                <div class="vm-accordion-header"
                     onclick="toggleAccordionItem('{item_id}')"
                     style="cursor: pointer; padding: 6px; padding-left: 33px; font-size: 14px; font-weight: normal; background-color: #F0F0F0; border: 1px solid #ddd;">
                    <span class="vm-accordion-toggle" id="{item_id}-toggle">▶</span>
                    {title}
                </div>
                <div class="vm-accordion-content"
                     id="{item_id}"
                     style="display: none; padding: 15px; border: 1px solid #ddd; border-top: none; background-color: #fff;">
                    {content}
                </div>
            </div>
        </div>

        <script>
        function toggleAccordionItem(itemId) {{
            const content = document.getElementById(itemId);
            const toggle = document.getElementById(itemId + '-toggle');

            if (content.style.display === 'none' || content.style.display === '') {{
                content.style.display = 'block';
                toggle.innerHTML = '▼';
            }} else {{
                content.style.display = 'none';
                toggle.innerHTML = '▶';
            }}
        }}
        </script>
        """


def _create_content_html(content: Dict[str, Any]) -> str:
    """Create HTML representation of a content block."""
    content_type = CONTENT_TYPE_MAP[content["content_type"]]

    if content["content_type"] not in ["metric", "test"]:
        return non_test_content_block_html.format(
            content_id=content["content_id"],
            content_type=content_type,
        )

    try:
        test_html = describe_test(test_id=content["content_id"], show=False)
        test_name = test_id_to_name(content["content_id"])
        # Wrap test/metric blocks in accordion with styling matching text blocks
        return _render_test_accordion(
            content=test_html,
            title=f"{content_type}: {test_name} ('{content['content_id']}')",
        )
    except LoadTestError:
        # Wrap failed test blocks in accordion for consistency
        failed_html = failed_content_block_html.format(test_id=content["content_id"])
        return _render_test_accordion(
            content=failed_html,
            title=f"{content_type}: Failed to load ('{content['content_id']}')",
        )


def _create_sub_section_html(
    sub_sections: List[Dict[str, Any]], section_number: str
) -> str:
    """Create HTML representation of a subsection."""
    if not sub_sections:
        return "<p>Empty Section</p>"

    accordion_items = []
    accordion_titles = []

    for i, section in enumerate(sub_sections):
        section_content = ""
        if section["sections"]:
            section_content = _create_sub_section_html(
                section["sections"], section_number=f"{section_number}.{i + 1}"
            )
        elif contents := section.get("contents", []):
            content_htmls = [_create_content_html(content) for content in contents]
            section_content = "".join(content_htmls)
        else:
            section_content = "<p>Empty Section</p>"

        accordion_items.append(section_content)
        accordion_titles.append(
            f"{section_number}.{i + 1}. {section['title']} ('{section['id']}')"
        )

    return StatefulHTMLRenderer.render_accordion(accordion_items, accordion_titles)


def _create_section_html(tree: List[Dict[str, Any]]) -> str:
    """Create HTML representation of sections."""
    accordion_items = []
    accordion_titles = []

    for i, section in enumerate(tree):
        section_content = ""
        if section.get("sections"):
            section_content = _create_sub_section_html(section["sections"], str(i + 1))

        if section.get("contents"):
            contents_html = "".join(
                [_create_content_html(content) for content in section["contents"]]
            )
            if section_content:
                section_content = section_content + contents_html
            else:
                section_content = contents_html

        if not section_content:
            section_content = "<p>Empty Section</p>"

        accordion_items.append(section_content)
        accordion_titles.append(f"{i + 1}. {section['title']} ('{section['id']}')")

    return StatefulHTMLRenderer.render_accordion(accordion_items, accordion_titles)


def preview_template(template: str) -> None:
    """Preview a template in Jupyter Notebook.

    Args:
        template (dict): The template to preview.
    """
    if not is_notebook():
        logger.warning("preview_template() only works in Jupyter Notebook")
        return

    html_content = StatefulHTMLRenderer.get_base_css()
    html_content += _create_section_html(
        _convert_sections_to_section_tree(template["sections"])
    )
    display(html_content)


def _get_section_tests(section: Dict[str, Any]) -> List[str]:
    """
    Get all the tests in a section and its subsections.

    Args:
        section: A dictionary representing a section.

    Returns:
        A list of tests in the section.
    """
    tests = [
        {
            "id": content["content_id"],
            "output_template": content.get("output_template"),
        }
        for content in section.get("contents", [])
        if content["content_type"] in ["metric", "test"]
    ]

    for sub_section in section["sections"]:
        tests.extend(_get_section_tests(sub_section))

    return tests


def _create_test_suite_section(section: Dict[str, Any]) -> Dict[str, Any]:
    """Create a section object for a test suite that contains the tests in a section
    in the template.

    Args:
        section: A section of a template (in tree form).

    Returns:
        A TestSuite section dict.
    """
    if section_tests := _get_section_tests(section):
        return {
            "section_id": section["id"],
            "section_description": section["title"],
            "section_tests": section_tests,
        }


def _create_template_test_suite(
    template: str, section: Optional[str] = None
) -> Type[TestSuite]:
    """
    Create and run a test suite from a template.

    Args:
        template: A valid flat template.
        section: The section of the template to run. Runs all sections if not provided.

    Returns:
        A dynamically-created TestSuite Class.
    """
    section_tree = _convert_sections_to_section_tree(
        sections=template["sections"],
        start_section_id=section,
    )

    # dynamically create a TestSuite class using `type` and populate it with the tests
    return type(
        f"{template['template_name'].title().replace(' ', '')}TestSuite",
        (TestSuite,),
        {
            "suite_id": template["template_id"],
            "tests": [
                section_dict
                for section in section_tree
                if (section_dict := _create_test_suite_section(section)) is not None
            ],
            "__doc__": template["description"],
        },
    )


def get_template_test_suite(template: str, section: Optional[str] = None) -> TestSuite:
    """Get a TestSuite instance containing all tests in a template.

    This function will collect all tests used in a template into a dynamically-created
    TestSuite object.

    Args:
        template: A valid flat template
        section: The section of the template to run (if not provided, run all sections)

    Returns:
        The TestSuite instance.
    """
    return _create_template_test_suite(template, section)()
