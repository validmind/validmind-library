#!/usr/bin/env python3
"""
sync_copyright.py

- Scans notebooks under repo_root/notebooks by default
- Only processes .ipynb files whose *filename does NOT* start with "_"
- Looks for a markdown cell containing: <!-- VALIDMIND COPYRIGHT -->
- If missing, appends the canonical copyright cell from repo_root/notebook/templates/_copyright.ipynb
- If present but different, replaces it with the canonical content

Usage examples:
  python scripts/sync_copyright.py
  python scripts/sync_copyright.py --dry-run
  python scripts/sync_copyright.py --check
  python scripts/sync_copyright.py --root path/to/other/notebooks
  python scripts/sync_copyright.py --copyright path/to/notebook/templates/_copyright.ipynb
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import nbformat

MARKER = "<!-- VALIDMIND COPYRIGHT -->"

# Assumes:
# repo/
#   notebooks/
#     templates/
#       _copyright.ipynb
#   scripts/
#     sync_copyright.py
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Result:
    path: Path
    status: str  # "unchanged" | "updated" | "appended" | "would-update" | "would-append" | "error"
    detail: str = ""


def _read_notebook(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(str(path), as_version=4)


def _write_notebook(path: Path, nb: nbformat.NotebookNode) -> None:
    nbformat.write(nb, str(path))


def _normalize_source(s: str) -> str:
    """
    Normalize for stable comparisons:
    - Convert Windows/Mac newlines to \n
    - Strip trailing whitespace on each line
    - Strip trailing newlines at end
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.rstrip("\n")


def _find_marked_markdown_cell_index(nb: nbformat.NotebookNode) -> Optional[int]:
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", "")
            if isinstance(src, str) and MARKER in src:
                return i
    return None


def _load_canonical_cell_source(copyright_nb_path: Path) -> str:
    if not copyright_nb_path.exists():
        raise FileNotFoundError(f"Copyright notebook not found: {copyright_nb_path}")

    nb = _read_notebook(copyright_nb_path)
    idx = _find_marked_markdown_cell_index(nb)
    if idx is None:
        raise ValueError(
            f"Could not find a markdown cell containing marker {MARKER!r} in {copyright_nb_path}"
        )

    canonical = nb.cells[idx].get("source", "")
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError(f"Canonical cell in {copyright_nb_path} is empty or invalid.")

    return canonical


def _iter_ipynb_files(root: Path, exclude_dirs: Tuple[str, ...]) -> Iterable[Path]:
    for path in root.rglob("*.ipynb"):
        # Skip typical noisy directories.
        if any(ex in path.parts for ex in exclude_dirs):
            continue

        # Only check notebooks whose filename does NOT start with "_"
        if path.name.startswith("_"):
            continue

        yield path


def process_notebook(
    nb_path: Path,
    canonical_source: str,
    *,
    dry_run: bool,
    check_only: bool,
) -> Result:
    try:
        nb = _read_notebook(nb_path)
    except Exception as e:
        return Result(nb_path, "error", f"failed to read: {e}")

    canonical_norm = _normalize_source(canonical_source)

    idx = _find_marked_markdown_cell_index(nb)
    if idx is None:
        if check_only:
            return Result(nb_path, "would-append", "marker not found")

        if not dry_run:
            nb.cells.append(nbformat.v4.new_markdown_cell(source=canonical_source))
            try:
                _write_notebook(nb_path, nb)
            except Exception as e:
                return Result(nb_path, "error", f"failed to write: {e}")

        return Result(nb_path, "appended" if not dry_run else "would-append", "marker not found")

    existing = nb.cells[idx].get("source", "")
    existing_norm = _normalize_source(existing if isinstance(existing, str) else "")

    if existing_norm == canonical_norm:
        return Result(nb_path, "unchanged", "already matches canonical")

    if check_only:
        return Result(nb_path, "would-update", "marker found but content differs")

    if not dry_run:
        nb.cells[idx]["source"] = canonical_source
        try:
            _write_notebook(nb_path, nb)
        except Exception as e:
            return Result(nb_path, "error", f"failed to write: {e}")

    return Result(
        nb_path,
        "updated" if not dry_run else "would-update",
        "marker found but content differed",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "notebooks",
        help="Root directory to scan (default: repo_root/notebooks)",
    )
    p.add_argument(
        "--copyright",
        type=Path,
        default=REPO_ROOT / "notebooks" / "templates" / "_copyright.ipynb",
        help="Notebook containing the canonical copyright cell (default: repo_root/notebooks/templates/_copyright.ipynb)",
    )
    p.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=[".ipynb_checkpoints", ".git", ".venv", "venv", "node_modules", "__pycache__"],
        help="Directory names to exclude anywhere in the path",
    )
    p.add_argument("--dry-run", action="store_true", help="Show what would change but don't write")
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any notebook would be updated/appended (no writes)",
    )
    p.add_argument(
        "--include-copyright-notebook",
        action="store_true",
        help="Also process the copyright notebook itself (default: skip it)",
    )
    args = p.parse_args()

    root = args.root.resolve()
    copyright_nb = args.copyright.resolve()

    try:
        canonical_source = _load_canonical_cell_source(copyright_nb)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    results: list[Result] = []
    for nb_path in _iter_ipynb_files(root, tuple(args.exclude_dirs)):
        if not args.include_copyright_notebook and nb_path.resolve() == copyright_nb:
            continue
        results.append(
            process_notebook(
                nb_path,
                canonical_source,
                dry_run=args.dry_run,
                check_only=args.check,
            )
        )

    # Print per-file changes/errors
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.status in {"appended", "updated", "would-append", "would-update", "error"}:
            print(f"{r.status:12} {r.path}  ({r.detail})")

    print("\nSummary:")
    for k in sorted(counts.keys()):
        print(f"  {k:12}: {counts[k]}")

    if args.check:
        # Any diff means failure (useful for CI)
        if counts.get("would-append", 0) or counts.get("would-update", 0) or counts.get("error", 0):
            return 1

    if counts.get("error", 0):
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
