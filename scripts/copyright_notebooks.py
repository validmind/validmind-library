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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import nbformat

MARKER = "<!-- VALIDMIND COPYRIGHT -->"
CELL_ID_PREFIX = "copyright-"

# Assumes:
# repo/
#   notebooks/
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
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.rstrip("\n")


def _new_copyright_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(
        source=source,
        id=f"{CELL_ID_PREFIX}{uuid.uuid4().hex}",
    )


def _find_marked_markdown_cell_index(nb: nbformat.NotebookNode) -> Optional[int]:
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", "")
            if isinstance(src, str) and MARKER in src:
                return i
    return None


def _load_canonical_cell_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Copyright notebook not found: {path}")

    nb = _read_notebook(path)
    idx = _find_marked_markdown_cell_index(nb)
    if idx is None:
        raise ValueError(
            f"Could not find a markdown cell containing marker {MARKER!r} in {path}"
        )

    canonical = nb.cells[idx].get("source", "")
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError(f"Canonical cell in {path} is empty or invalid.")

    return canonical


def _iter_ipynb_files(root: Path, exclude_dirs: Tuple[str, ...]) -> Iterable[Path]:
    for path in root.rglob("*.ipynb"):
        if any(ex in path.parts for ex in exclude_dirs):
            continue
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

    # ---------- Missing cell ----------
    if idx is None:
        if check_only:
            return Result(nb_path, "would-append", "marker not found")

        if not dry_run:
            nb.cells.append(_new_copyright_cell(canonical_source))
            try:
                _write_notebook(nb_path, nb)
            except Exception as e:
                return Result(nb_path, "error", f"failed to write: {e}")

        return Result(nb_path, "appended" if not dry_run else "would-append", "marker not found")

    # ---------- Existing cell ----------
    cell = nb.cells[idx]
    existing_source = cell.get("source", "")
    existing_norm = _normalize_source(existing_source if isinstance(existing_source, str) else "")

    id_ok = isinstance(cell.get("id"), str) and cell["id"].startswith(CELL_ID_PREFIX)

    if existing_norm == canonical_norm and id_ok:
        return Result(nb_path, "unchanged", "already matches canonical")

    if check_only:
        return Result(nb_path, "would-update", "content or id differs")

    if not dry_run:
        nb.cells[idx] = _new_copyright_cell(canonical_source)
        try:
            _write_notebook(nb_path, nb)
        except Exception as e:
            return Result(nb_path, "error", f"failed to write: {e}")

    return Result(
        nb_path,
        "updated" if not dry_run else "would-update",
        "content or id differed",
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
        help="Canonical copyright notebook",
    )
    p.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=[".ipynb_checkpoints", ".git", ".venv", "venv", "node_modules", "__pycache__"],
        help="Directory names to exclude anywhere in the path",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    root = args.root.resolve()
    copyright_nb = args.copyright.resolve()

    try:
        canonical_source = _load_canonical_cell_source(copyright_nb)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    counts: dict[str, int] = {}
    for nb_path in _iter_ipynb_files(root, tuple(args.exclude_dirs)):
        result = process_notebook(
            nb_path,
            canonical_source,
            dry_run=args.dry_run,
            check_only=args.check,
        )
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status not in {"unchanged"}:
            print(f"{result.status:12} {result.path} ({result.detail})")

    print("\nSummary:")
    for k in sorted(counts):
        print(f"  {k:12}: {counts[k]}")

    if args.check and any(k in counts for k in ("would-append", "would-update", "error")):
        return 1
    if counts.get("error"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
