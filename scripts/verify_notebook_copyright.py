# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution or use of this software is strictly prohibited.
# Please refer to the LICENSE file in the root directory of this repository
# for more information.
#
# Copyright © 2023 ValidMind Inc. All rights reserved.

"""
This script verifies that all notebooks under a directory have the
ValidMind copyright cell.

How to use:
    poetry run python scripts/verify_notebook_copyright.py

Notes:
- Checks for a markdown cell containing: <!-- VALIDMIND COPYRIGHT -->
- Compares that cell's content to notebooks/templates/_copyright.ipynb
- Only checks .ipynb files whose filename does NOT start with "_"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import nbformat

MARKER = "<!-- VALIDMIND COPYRIGHT -->"
CELL_ID_PREFIX = "copyright-"


def normalize_source(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.rstrip("\n")


def find_marked_markdown_cell_index(nb: nbformat.NotebookNode) -> Optional[int]:
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", "")
            if isinstance(src, str) and MARKER in src:
                return i
    return None


def load_canonical_cell_source(copyright_nb_path: Path) -> str:
    if not copyright_nb_path.exists():
        raise FileNotFoundError(f"Canonical copyright notebook not found: {copyright_nb_path}")

    nb = nbformat.read(str(copyright_nb_path), as_version=4)
    idx = find_marked_markdown_cell_index(nb)
    if idx is None:
        raise ValueError(
            f"Could not find a markdown cell containing marker {MARKER!r} in {copyright_nb_path}"
        )

    canonical = nb.cells[idx].get("source", "")
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError(f"Canonical cell in {copyright_nb_path} is empty or invalid.")

    return canonical


def main() -> int:
    repo_root = Path(os.getcwd()).resolve()

    # Align with the existing script style:
    # - default paths rooted at cwd (repo root when run via poetry/make)
    notebooks_dir = repo_root / "notebooks"
    copyright_nb_path = repo_root / "notebooks" / "templates" / "_copyright.ipynb"

    try:
        canonical_source = load_canonical_cell_source(copyright_nb_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    canonical_norm = normalize_source(canonical_source)

    errors: list[str] = []

    for root, dirs, files in os.walk(notebooks_dir):
        # Skip noisy directories
        dirs[:] = [d for d in dirs if d not in {".ipynb_checkpoints", ".git", "__pycache__"}]

        for file in files:
            if not file.endswith(".ipynb"):
                continue
            if file.startswith("_"):
                continue

            nb_path = Path(root) / file

            try:
                nb = nbformat.read(str(nb_path), as_version=4)
            except Exception as e:
                errors.append(f"Notebook {nb_path} could not be read: {e}")
                continue

            idx = find_marked_markdown_cell_index(nb)
            if idx is None:
                errors.append(f"Notebook {nb_path} is missing the copyright cell.")
                continue

            cell = nb.cells[idx]
            src = cell.get("source", "")
            src_norm = normalize_source(src if isinstance(src, str) else "")

            if src_norm != canonical_norm:
                errors.append(
                    f"Notebook {nb_path} has a copyright cell, but its content does not match canonical."
                )

            cell_id = cell.get("id")
            if not (isinstance(cell_id, str) and cell_id.startswith(CELL_ID_PREFIX)):
                errors.append(
                    f"Notebook {nb_path} copyright cell is missing a valid id (expected prefix '{CELL_ID_PREFIX}')."
                )

    if errors:
        print("\n".join(errors))
        print("\nPlease fix the errors above by running `make copyright`")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
