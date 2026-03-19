"""Tests for md_to_html in validmind.utils."""

from validmind.utils import md_to_html


# ---------------------------------------------------------------------------
# Basic Markdown conversion
# ---------------------------------------------------------------------------


def test_heading_converted():
    result = md_to_html("# Heading")
    assert "<h1>Heading</h1>" in result


def test_paragraph_preserved():
    result = md_to_html("Hello world.")
    assert "<p>Hello world.</p>" in result


def test_bold_converted():
    result = md_to_html("**bold**")
    assert "<strong>bold</strong>" in result


def test_italic_converted():
    result = md_to_html("*italic*")
    assert "<em>italic</em>" in result


def test_empty_string():
    result = md_to_html("")
    assert result.strip() == ""


# ---------------------------------------------------------------------------
# Table conversion — CKEditor compatibility
# ---------------------------------------------------------------------------


def test_table_wrapped_in_figure():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = md_to_html(md)
    assert '<figure class="table"><table>' in result
    assert "</table></figure>" in result


def test_table_cells_rendered():
    md = "| Name | Score |\n|------|-------|\n| Alice | 95 |"
    result = md_to_html(md)
    assert "Alice" in result
    assert "95" in result


def test_table_header_rendered():
    md = "| H1 | H2 |\n|----|----|\n| a | b |"
    result = md_to_html(md)
    assert "<th>" in result


def test_multiple_tables_both_wrapped():
    md = "| A |\n|---|\n| 1 |\n\nText\n\n| B |\n|---|\n| 2 |"
    result = md_to_html(md)
    assert result.count('<figure class="table">') == 2


# ---------------------------------------------------------------------------
# Math conversion (mathml=False — no math/tex rewriting)
# ---------------------------------------------------------------------------


def test_inline_math_without_mathml():
    """Without mathml, mistune wraps inline math in <span class="math">."""
    result = md_to_html("Value $x + y$ here.")
    assert "x + y" in result


def test_block_math_without_mathml():
    """Without mathml, mistune wraps block math in <div class="math">."""
    result = md_to_html("$$E = mc^2$$")
    assert "E = mc^2" in result


# ---------------------------------------------------------------------------
# Math conversion (mathml=True — rewrite to math/tex script tags)
# ---------------------------------------------------------------------------


def test_block_math_converted_to_script_tag():
    # Mistune requires $$ on its own line for block-level math
    result = md_to_html("$$\nE = mc^2\n$$", mathml=True)
    assert '<script type="math/tex; mode=display">' in result
    assert "E = mc^2" in result


def test_inline_math_converted_to_script_tag():
    result = md_to_html("Value $x + y$ here.", mathml=True)
    assert '<script type="math/tex">x + y</script>' in result


def test_multiple_inline_math():
    result = md_to_html("$a$ and $b$", mathml=True)
    assert result.count('<script type="math/tex">') == 2


# ---------------------------------------------------------------------------
# Code blocks
# ---------------------------------------------------------------------------


def test_fenced_code_block():
    md = "```python\ndef hello():\n    pass\n```"
    result = md_to_html(md)
    assert "code" in result
    assert "hello" in result


def test_inline_code():
    result = md_to_html("Use `len()` to count.")
    assert "<code>" in result
    assert "len()" in result


# ---------------------------------------------------------------------------
# Mixed content
# ---------------------------------------------------------------------------


def test_mixed_text_table_and_math():
    md = "Formula $x^2$.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nDone."
    result = md_to_html(md, mathml=True)
    assert '<script type="math/tex">x^2</script>' in result
    assert '<figure class="table">' in result
    assert "<p>Done.</p>" in result
