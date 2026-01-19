from __future__ import annotations
import nbformat
import os
import uuid
import subprocess
import json
from typing import Callable, Dict, Iterable, Optional, Set, Tuple

def ensure_ids(notebook):
    """Ensure every cell in the notebook has a unique id."""
    for cell in notebook.cells:
        if "id" not in cell:
            cell["id"] = str(uuid.uuid4())
    return notebook

def create_notebook():
    """Creates a new Jupyter Notebook file by asking the user for a filename and opens it in VS Code."""
    filename = input("Enter the name for the new notebook (without .ipynb extension): ").strip()
    if not filename:
        print("Filename cannot be empty, file not created")
        return

    if not filename.endswith(".ipynb"):
        filename += ".ipynb"

    current_dir = os.path.dirname(__file__)
    directory = os.path.join(current_dir, "..", "code_sharing")

    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)

    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    }

    notebook = ensure_ids(notebook)

    try:
        with open(filepath, "w") as f:
            nbformat.write(notebook, f)
        print(f"Created '{filepath}'")

        subprocess.run(["code", filepath], check=True)
    except Exception as e:
        print(f"Error creating or opening notebook: {e}")

    return filepath

def set_title(filepath):
    """Adds a markdown cell with a h1 title to the specified notebook."""
    if not os.path.exists(filepath):
        print("The specified notebook file does not exist")
        return

    try:
        with open(filepath, "r") as f:
            notebook = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    title = input("Enter the title for the notebook: ").strip()
    if not title:
        print("No title inputted, skipped insertion")
        return

    markdown_cell = nbformat.v4.new_markdown_cell(f"# {title}")
    notebook.cells.insert(0, markdown_cell)

    notebook = ensure_ids(notebook)

    try:
        with open(filepath, "w") as f:
            nbformat.write(notebook, f)
        print(f"Set title for '{filepath}': '{title}'")
    except Exception as e:
        print(f"Error updating notebook: {e}")

def add_about(filepath):
    """Appends the contents of '_about-validmind.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "_about-validmind.ipynb")

    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    user_input = input("Do you want to include information about ValidMind? (yes/no): ").strip().lower()
    if user_input not in ("yes", "y"):
        print("Skipping appending '_about-validmind.ipynb'")
        return

    try:
        with open(filepath, "r") as target_file:
            target_notebook = nbformat.read(target_file, as_version=4)

        with open(source_notebook_path, "r") as source_file:
            source_notebook = nbformat.read(source_file, as_version=4)
    except Exception as e:
        print(f"Error reading notebooks: {e}")
        return

    for cell in source_notebook.cells:
        original_id = cell.get("id", f"cell-{uuid.uuid4()}")
        new_id = f"{original_id}-{uuid.uuid4()}"
        cell["id"] = new_id

    target_notebook.cells.extend(source_notebook.cells)
    target_notebook = ensure_ids(target_notebook)

    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"'_about-validmind.ipynb' appended to '{filepath}'")
    except Exception as e:
        print(f"Error appending notebooks: {e}")



# Prefix -> allowed cell types to remove (None/empty => any type)
DEFAULT_SKIP_PREFIX_RULES: Dict[str, Optional[Iterable[str]]] = {
    "install-template": {"markdown"},
    "install-preview": {"markdown"},
    "install-preview-template": {"code"},
}


def _normalize_cell_type(ctype: Optional[str]) -> str:
    return (ctype or "").lower()


def _first_line(text: str) -> str:
    # Normalize common leading comment markers in first line for better matching
    line = text.splitlines()[0].lstrip("\ufeff ")  # strip BOM/space
    for prefix in ("#", "//", "--", "/*", "<!--"):
        if line.startswith(prefix):
            return line[len(prefix) :].lstrip()
    return line


def _get_id_candidates(cell: dict) -> Iterable[str]:
    """Collect likely identifiers for prefix matching.
    - cell.id (v4.5+)
    - cell.metadata.id (common)
    - any string value directly under metadata (shallow)
    - first line of source (string or first element of list)
    """
    # id
    cid = cell.get("id")
    if isinstance(cid, str) and cid:
        yield cid
    # metadata
    md = cell.get("metadata") or {}
    if isinstance(md, dict):
        mid = md.get("id")
        if isinstance(mid, str) and mid:
            yield mid
        for v in md.values():
            if isinstance(v, str) and v:
                yield v
    # source first line
    src = cell.get("source")
    if isinstance(src, str) and src:
        yield _first_line(src)
    elif isinstance(src, list) and src and isinstance(src[0], str):
        yield _first_line(src[0])


def _remove_cells_by_prefix_rules(
    nb: dict, prefix_rules: Dict[str, Optional[Iterable[str]]]
) -> Tuple[dict, int]:
    removed = 0

    def match(cell: dict) -> bool:
        ctype = _normalize_cell_type(cell.get("cell_type"))
        cands = list(_get_id_candidates(cell))
        for prefix, allowed_types in prefix_rules.items():
            allowed: Optional[Set[str]] = (
                {t.lower() for t in allowed_types} if allowed_types else None
            )
            if allowed is not None and ctype not in allowed:
                continue
            for s in cands:
                if isinstance(s, str) and s.startswith(prefix):
                    return True
        return False

    # nbformat v4
    if isinstance(nb, dict) and isinstance(nb.get("cells"), list):
        new_cells = []
        for c in nb["cells"]:
            if match(c):
                removed += 1
            else:
                new_cells.append(c)
        if removed:
            nb = {**nb, "cells": new_cells}
        return nb, removed

    # nbformat v3 (worksheets)
    if isinstance(nb.get("worksheets"), list) and nb["worksheets"]:
        ws = nb["worksheets"][0]
        cells = ws.get("cells") or []
        new_cells = []
        for c in cells:
            if match(c):
                removed += 1
            else:
                new_cells.append(c)
        if removed:
            ws = {**ws, "cells": new_cells}
            nb = {**nb, "worksheets": [ws] + nb["worksheets"][1:]}
        return nb, removed

    return nb, removed


def replace_variables(
    filepath: str,
    *,
    prefix_rules: Optional[Dict[str, Optional[Iterable[str]]]] = None,
    always_prompt: bool = True,
    template_value: Optional[str] = None,
    use_case_value: Optional[str] = None,
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> None:
    """Replace variables in a notebook-like JSON file and optionally delete cells by id prefix.

    Behavior:
    - Prompts for {template} replacement. If user types "SKIP", deletes cells
      according to `prefix_rules` (defaults to DEFAULT_SKIP_PREFIX_RULES) regardless of
      whether the placeholder exists in the file.
    - If a non-empty value is given, replaces {template} with that value.
    - Also prompts for and replaces {use-case} if present (or when value provided).

    Set `always_prompt=True` to prompt even if "{template}" is not present in the file content.

    For non-interactive usage, pass `template_value` and/or `use_case_value`.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        rules = prefix_rules or DEFAULT_SKIP_PREFIX_RULES
        replacements_made = False

        # Decide whether to process template
        should_prompt = always_prompt or ("{template}" in content)

        user_in: Optional[str] = template_value
        if should_prompt and user_in is None:
            user_in = input_func(
                "Enter a value to replace {template} or type SKIP to exclude the template selection: "
            ).strip()
        elif template_value is not None:
            user_in = template_value.strip()

        if should_prompt and user_in is not None:
            if user_in.upper() == "SKIP":
                # Attempt structured removal; if JSON fails, leave content unchanged
                try:
                    nb = json.loads(content)
                    nb, removed = _remove_cells_by_prefix_rules(nb, rules)
                    if removed:
                        content = json.dumps(nb, ensure_ascii=False, indent=1)
                        print_func(
                            f"Skipped template selection; removed {removed} cell(s) matching prefixes {list(rules.keys())}."
                        )
                    else:
                        print_func("Skipped template selection; no matching cells found.")
                except json.JSONDecodeError:
                    print_func(
                        "Skipped template selection, but file is not valid notebook JSON; no cells removed."
                    )
            elif user_in:
                content = content.replace("{template}", user_in)
                print_func(f"Template: {user_in}")
                replacements_made = True
            else:
                print_func("No value entered for {template}, skipping replacement")

        # Handle {use-case}
        if "{use-case}" in content or use_case_value is not None:
            uc = use_case_value
            if uc is None:
                uc = input_func(
                    "Enter a value to replace {use-case} or type SKIP to exclude the template selection: "
                ).strip()
            else:
                uc = uc.strip()

            if uc.upper() == "SKIP":
                # Remove markdown cells whose id starts with 'install-usecase'
                try:
                    nb = json.loads(content)
                    usecase_rules = {"install-usecase": {"markdown"}}
                    nb, removed = _remove_cells_by_prefix_rules(nb, usecase_rules)
                    if removed:
                        content = json.dumps(nb, ensure_ascii=False, indent=1)
                        print_func(
                            "Skipped use-case selection; removed "
                            f"{removed} cell(s) matching prefixes ['install-usecase']."
                        )
                    else:
                        print_func("Skipped use-case selection; no matching cells found.")
                except json.JSONDecodeError:
                    print_func(
                        "Skipped use-case selection, but file is not valid notebook JSON; no cells removed."
                    )
            elif uc:
                content = content.replace("{use-case}", uc)
                print_func(f"Use case: {uc}")
                replacements_made = True
            else:
                print_func("No value entered for {use-case}, skipping replacement")

        with open(filepath, "w", encoding="utf-8") as f:
            if not content.endswith("\n"):
                content += "\n"
            f.write(content)

        if replacements_made:
            print_func(f"Replaced template and/or use case variables in '{filepath}'")

    except Exception as e:
        print_func(f"Error replacing variables in file: {e}")

def add_install(filepath):
    """Appends the contents of '_install-initialize-validmind.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "_install-initialize-validmind.ipynb")

    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    user_input = input("Do you want to include installation and initialization instructions? (yes/no): ").strip().lower()
    if user_input not in ("yes", "y"):
        print("Skipping appending '_install-initialize-validmind.ipynb'")
        return

    try:
        with open(filepath, "r") as target_file:
            target_notebook = nbformat.read(target_file, as_version=4)

        with open(source_notebook_path, "r") as source_file:
            source_notebook = nbformat.read(source_file, as_version=4)
    except Exception as e:
        print(f"Error reading notebooks: {e}")
        return

    for cell in source_notebook.cells:
        original_id = cell.get("id", f"cell-{uuid.uuid4()}")
        new_id = f"{original_id}-{uuid.uuid4()}"
        cell["id"] = new_id

    target_notebook.cells.extend(source_notebook.cells)
    target_notebook = ensure_ids(target_notebook)

    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"'_install-initialize-validmind.ipynb' appended to '{filepath}'")
    except Exception as e:
        print(f"Error appending notebooks: {e}")

    replace_variables(filepath)

def next_steps(filepath):
    """Appends the contents of '_next-steps.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "_next-steps.ipynb")

    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    user_input = input("Do you want to include next steps? (yes/no): ").strip().lower()
    if user_input not in ("yes", "y"):
        print("Skipping appending '_next-steps.ipynb'")
        return

    try:
        with open(filepath, "r") as target_file:
            target_notebook = nbformat.read(target_file, as_version=4)

        with open(source_notebook_path, "r") as source_file:
            source_notebook = nbformat.read(source_file, as_version=4)
    except Exception as e:
        print(f"Error reading notebooks: {e}")
        return

    for cell in source_notebook.cells:
        original_id = cell.get("id", f"cell-{uuid.uuid4()}")
        new_id = f"{original_id}-{uuid.uuid4()}"
        cell["id"] = new_id

    target_notebook.cells.extend(source_notebook.cells)
    target_notebook = ensure_ids(target_notebook)

    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"'_next-steps.ipynb' appended to '{filepath}'")
    except Exception as e:
        print(f"Error appending notebooks: {e}")

def add_upgrade(filepath):
    """Appends the contents of '_upgrade-validmind.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "_upgrade-validmind.ipynb")

    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    user_input = input("Do you want to include information about upgrading ValidMind? (yes/no): ").strip().lower()
    if user_input not in ("yes", "y"):
        print("Skipping appending '_upgrade-validmind.ipynb'")
        return

    try:
        with open(filepath, "r") as target_file:
            target_notebook = nbformat.read(target_file, as_version=4)

        with open(source_notebook_path, "r") as source_file:
            source_notebook = nbformat.read(source_file, as_version=4)
    except Exception as e:
        print(f"Error reading notebooks: {e}")
        return

    for cell in source_notebook.cells:
        original_id = cell.get("id", f"cell-{uuid.uuid4()}")
        new_id = f"{original_id}-{uuid.uuid4()}"
        cell["id"] = new_id

    target_notebook.cells.extend(source_notebook.cells)
    target_notebook = ensure_ids(target_notebook)

    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"'_upgrade-validmind.ipynb' appended to '{filepath}'")
    except Exception as e:
        print(f"Error appending notebooks: {e}")

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    filepath = create_notebook()
    if filepath:
        set_title(filepath)
        add_about(filepath)
        add_install(filepath)
        next_steps(filepath)
        add_upgrade(filepath)
