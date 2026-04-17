"""Grep utility for the Localizer agent.

Searches actual repository file content for keyword matches.  Graph scoring
only looks at node names and paths; grep catches cases where the relevant
code has a keyword only in a comment, macro, or string literal that
tree-sitter did not index as a function name.

Design
------
- Pure Python `re` — no subprocess, portable across Windows/Linux
- Whole-word matching: wraps each keyword in \\b so "compress" does not
  match "decompress"
- Case-insensitive search
- Skips the same vendor/build directories as graph/builder.py
- Handles unreadable / binary files gracefully
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Directories to skip — kept in sync with graph/builder.py SKIP_DIRS
SKIP_DIRS: set[str] = {
    "vendor", "deps", "contrib", "third_party", "third-party",
    "external", "node_modules", ".git", "build", "cmake-build",
}

# File extensions to search
_C_EXTENSIONS: tuple[str, ...] = (".c", ".h")


@dataclass
class GrepHit:
    """One line that matched a keyword search."""
    file_path: str    # repo-relative POSIX path, e.g. "src/jv.c"
    line_no: int      # 1-indexed
    line_text: str    # stripped source line (≤ 200 chars)
    keyword: str      # which keyword triggered this hit


def _should_skip(path: Path, repo_root: Path) -> bool:
    """Return True if any path component is in SKIP_DIRS."""
    try:
        rel_parts = path.relative_to(repo_root).parts
    except ValueError:
        return True
    return any(p.lower() in SKIP_DIRS for p in rel_parts)


def _compile_pattern(keyword: str) -> re.Pattern:
    """Compile a case-insensitive whole-word regex for a keyword.

    For identifiers with underscores (e.g. ``jv_parse``), ``\\b`` boundaries
    work correctly.  For keywords that start/end with ``_`` we fall back to a
    lookaround so the match still fires.
    """
    escaped = re.escape(keyword)
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


def grep_repo(
    repo_root: Path,
    keywords: list[str],
    file_extensions: tuple[str, ...] = _C_EXTENSIONS,
    max_hits_per_keyword: int = 60,
    skip_dirs: set[str] = SKIP_DIRS,
) -> dict[str, list[GrepHit]]:
    """Search repository files for keyword matches.

    Args:
        repo_root:           Root of the checked-out repository.
        keywords:            Search terms (C identifier style).
        file_extensions:     File types to scan.
        max_hits_per_keyword: Stop collecting hits for a keyword after this
                              many matches (prevents flooding on generic terms).
        skip_dirs:           Directory names to skip.

    Returns:
        ``{rel_file_path: [GrepHit, ...]}`` — only files with ≥ 1 hit.
        Hits within each file are deduplicated by line number.
    """
    if not keywords:
        return {}

    repo_root = Path(repo_root)
    patterns = {kw: _compile_pattern(kw) for kw in keywords}

    # hits_per_kw tracks the global count to enforce max_hits_per_keyword
    hits_per_kw: dict[str, int] = {kw: 0 for kw in keywords}

    # result: file_path → set of (line_no, line_text, keyword)
    # We use a dict of list[GrepHit] keyed by file
    result: dict[str, dict[int, GrepHit]] = {}  # file → {line_no: GrepHit}

    # Collect all target files
    target_files: list[Path] = []
    for ext in file_extensions:
        for p in repo_root.rglob(f"*{ext}"):
            if not _should_skip(p, repo_root):
                target_files.append(p)

    for fpath in target_files:
        rel = fpath.relative_to(repo_root).as_posix()

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lines = text.splitlines()
        file_hits: dict[int, GrepHit] = {}

        for lineno, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            for kw, pat in patterns.items():
                if hits_per_kw[kw] >= max_hits_per_keyword:
                    continue
                if pat.search(line):
                    if lineno not in file_hits:
                        # First keyword win per line — record the match
                        file_hits[lineno] = GrepHit(
                            file_path=rel,
                            line_no=lineno,
                            line_text=line[:200],
                            keyword=kw,
                        )
                    hits_per_kw[kw] += 1

        if file_hits:
            result[rel] = file_hits

    # Convert inner dicts → sorted lists
    return {
        fp: sorted(hits.values(), key=lambda h: h.line_no)
        for fp, hits in result.items()
    }


def hit_density(
    grep_hits: dict[str, list[GrepHit]],
) -> dict[str, float]:
    """Compute per-file hit density normalised to [0, 1].

    Returns ``{file_path: density}`` where the file with the most hits
    gets 1.0 and others are scaled proportionally.
    """
    if not grep_hits:
        return {}
    counts = {fp: len(hits) for fp, hits in grep_hits.items()}
    max_count = max(counts.values()) or 1
    return {fp: c / max_count for fp, c in counts.items()}
