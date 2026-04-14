"""File reading and function snippet extraction for the Localizer agent.

Reads actual source files from the checked-out repository and extracts code
snippets corresponding to specific function line ranges taken from the graph's
function nodes.

Design
------
- 1-indexed line numbers (matching graph node start_line / end_line fields)
- Snippets include a small context padding above and below the function body
  so that surrounding macro definitions, typedefs, and comments are visible
- Snippets are capped at MAX_SNIPPET_LINES to avoid token bloat for downstream
  agents
- Missing / unreadable files return None / empty string gracefully
"""

from __future__ import annotations

from pathlib import Path

from graph.model import DepGraph

# Maximum lines per extracted snippet (prevents flooding Diagnostician prompt)
MAX_SNIPPET_LINES: int = 150
# Lines of surrounding context to include above/below function body
DEFAULT_CONTEXT_LINES: int = 3


def read_file(repo_root: Path, rel_path: str) -> str | None:
    """Read a source file from disk.

    Args:
        repo_root: Root of the checked-out repository.
        rel_path:  Repo-relative POSIX path, e.g. ``"src/jv.c"``.

    Returns:
        File text as a string, or ``None`` if the file cannot be read.
    """
    full_path = Path(repo_root) / rel_path
    try:
        return full_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def extract_snippet(
    file_text: str,
    start_line: int,
    end_line: int,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> str:
    """Extract a line range from ``file_text`` with context padding.

    Args:
        file_text:     Full file content as a single string.
        start_line:    First line of the region (1-indexed, inclusive).
        end_line:      Last line of the region (1-indexed, inclusive).
        context_lines: Extra lines to include above and below the region.

    Returns:
        The extracted text block, preserving original indentation.
        Includes a ``# L<start>`` header comment for orientation.
    """
    lines = file_text.splitlines()
    total = len(lines)

    # Convert to 0-indexed, clamp to file bounds
    lo = max(0, start_line - 1 - context_lines)
    hi = min(total, end_line + context_lines)

    # Cap total length
    if hi - lo > MAX_SNIPPET_LINES:
        hi = lo + MAX_SNIPPET_LINES

    snippet_lines = lines[lo:hi]

    # Add a concise location header so the Diagnostician can orient itself
    header = f"// --- {start_line}–{end_line} (context ±{context_lines}) ---"
    return header + "\n" + "\n".join(snippet_lines)


def extract_function_snippets(
    repo_root: Path,
    graph: DepGraph,
    func_node_ids: list[str],
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> dict[str, str]:
    """Extract code snippets for a list of function node IDs.

    Looks up start_line / end_line from the graph, reads the corresponding
    file, and returns the annotated snippet.

    Args:
        repo_root:      Root of the checked-out repository.
        graph:          The dependency graph (provides node metadata).
        func_node_ids:  List of function node IDs, e.g. ``["src/jv.c::jv_parse"]``.
        context_lines:  Lines of context above/below each function.

    Returns:
        ``{func_node_id: snippet_text}`` — missing nodes / files are skipped.
    """
    # Cache file text to avoid re-reading the same file multiple times
    file_cache: dict[str, str | None] = {}
    snippets: dict[str, str] = {}

    for nid in func_node_ids:
        node = graph.nodes.get(nid)
        if node is None or node.get("kind") != "function":
            continue

        file_path: str = node["path"]
        start: int = node.get("start_line", 1)
        end: int = node.get("end_line", start)

        if file_path not in file_cache:
            file_cache[file_path] = read_file(repo_root, file_path)

        text = file_cache[file_path]
        if text is None:
            continue

        snippets[nid] = extract_snippet(text, start, end, context_lines)

    return snippets


def read_top_files(
    repo_root: Path,
    file_paths: list[str],
) -> dict[str, str]:
    """Read multiple files and return their full text.

    Args:
        repo_root:   Root of the checked-out repository.
        file_paths:  List of repo-relative POSIX paths.

    Returns:
        ``{file_path: text}`` — files that cannot be read are omitted.
    """
    result = {}
    for rel in file_paths:
        text = read_file(repo_root, rel)
        if text is not None:
            result[rel] = text
    return result
