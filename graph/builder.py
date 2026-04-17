"""Build a code dependency graph from a C repository using tree-sitter."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import tree_sitter_c as tsc
from tree_sitter import Language, Parser

from graph.model import DepGraph

C_LANG = Language(tsc.language())

# Directories to skip during parsing
SKIP_DIRS = {
    "vendor", "deps", "contrib", "third_party", "third-party",
    "external", "node_modules", ".git", "build", "cmake-build",
}

# Heuristic: test file detection
_TEST_PATH_RE = re.compile(r"(^|/)tests?/|test_|_test\.c$", re.IGNORECASE)


def _is_test_file(path: str) -> bool:
    return bool(_TEST_PATH_RE.search(path))


def _should_skip(path: Path, repo_root: Path) -> bool:
    rel_parts = path.relative_to(repo_root).parts
    return any(p.lower() in SKIP_DIRS for p in rel_parts)


# ── Tree-sitter extraction helpers ────────────────────────────────


def _extract_functions(root_node) -> list[dict]:
    """Extract function definitions from AST root."""
    funcs = []
    for child in root_node.children:
        if child.type != "function_definition":
            continue
        decl = child.child_by_field_name("declarator")
        # Handle pointer declarators: int *foo(...)
        while decl and decl.type == "pointer_declarator":
            decl = decl.child_by_field_name("declarator")
        if not decl or decl.type != "function_declarator":
            continue
        name_node = decl.child_by_field_name("declarator")
        if not name_node:
            continue
        funcs.append({
            "name": name_node.text.decode(),
            "start_line": child.start_point[0] + 1,
            "end_line": child.end_point[0] + 1,
            "start_byte": child.start_byte,
            "end_byte": child.end_byte,
        })
    return funcs


def _extract_includes(root_node) -> list[str]:
    """Extract #include paths (quoted includes only, skip system headers)."""
    includes = []
    for child in root_node.children:
        if child.type != "preproc_include":
            continue
        path_node = child.child_by_field_name("path")
        if not path_node:
            continue
        raw = path_node.text.decode()
        # Only local includes: "foo.h", not <stdio.h>
        if raw.startswith('"') and raw.endswith('"'):
            includes.append(raw.strip('"'))
    return includes


def _extract_calls(node) -> list[str]:
    """Recursively find all function call names under a node."""
    calls = []
    if node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn and fn.type == "identifier":
            calls.append(fn.text.decode())
    for child in node.children:
        calls.extend(_extract_calls(child))
    return calls


def _extract_struct_defs(root_node) -> list[str]:
    """Extract struct/type names defined at file scope."""
    names = []
    for child in root_node.children:
        # typedef struct { ... } Name;
        if child.type == "type_definition":
            for sub in child.children:
                if sub.type == "struct_specifier":
                    name_node = sub.child_by_field_name("name")
                    body = sub.child_by_field_name("body")
                    if name_node and body:
                        names.append(name_node.text.decode())
            # Also get the typedef alias name (last identifier)
            decl = child.child_by_field_name("declarator")
            if decl and decl.type == "type_identifier":
                names.append(decl.text.decode())
        # struct Foo { ... }; at file scope
        elif child.type == "struct_specifier":
            name_node = child.child_by_field_name("name")
            body = child.child_by_field_name("body")
            if name_node and body:
                names.append(name_node.text.decode())
    return names


def _extract_struct_refs(root_node) -> set[str]:
    """Extract struct/type names referenced (used) in a file."""
    refs = set()
    _walk_struct_refs(root_node, refs)
    return refs


def _walk_struct_refs(node, refs: set[str]) -> None:
    if node.type == "struct_specifier":
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if name_node and not body:  # usage, not definition
            refs.add(name_node.text.decode())
    for child in node.children:
        _walk_struct_refs(child, refs)


# ── Main builder ──────────────────────────────────────────────────


def build_graph(repo_root: str | Path) -> DepGraph:
    """Parse a C repository and build a dependency graph.

    Args:
        repo_root: Path to the repository root directory.

    Returns:
        A populated DepGraph instance.
    """
    repo_root = Path(repo_root)
    graph = DepGraph()
    parser = Parser(C_LANG)

    # Collect all C/H files
    c_files: list[Path] = []
    for ext in ("*.c", "*.h"):
        for p in repo_root.rglob(ext):
            if not _should_skip(p, repo_root):
                c_files.append(p)

    # Global indexes for cross-file resolution
    func_index: dict[str, list[str]] = defaultdict(list)  # name -> [node_ids]
    struct_def_index: dict[str, str] = {}  # struct_name -> file_path
    header_index: dict[str, str] = {}  # basename -> rel_path

    # Per-file parsed data (stored for second pass)
    file_data: dict[str, dict] = {}

    # ── First pass: parse all files, build indexes ────────────────

    for fpath in c_files:
        rel = fpath.relative_to(repo_root).as_posix()

        # Index header by basename for include resolution
        if fpath.suffix == ".h":
            header_index[fpath.name] = rel

        try:
            source = fpath.read_bytes()
        except (OSError, UnicodeDecodeError):
            continue

        tree = parser.parse(source)
        root = tree.root_node

        # Add file node
        graph.add_file_node(rel)

        # Extract data
        funcs = _extract_functions(root)
        includes = _extract_includes(root)
        struct_defs = _extract_struct_defs(root)
        struct_refs = _extract_struct_refs(root)
        calls_by_func: dict[str, list[str]] = {}

        # Add function nodes and build func_index
        for f in funcs:
            nid = graph.add_func_node(rel, f["name"], f["start_line"], f["end_line"])
            func_index[f["name"]].append(nid)

            # Extract calls within this function's body
            # Walk the function_definition node for call_expression
            # We re-walk the AST for the function body byte range
            body_calls = []
            for child in root.children:
                if (child.type == "function_definition"
                        and child.start_byte == f["start_byte"]):
                    body = child.child_by_field_name("body")
                    if body:
                        body_calls = _extract_calls(body)
                    break
            calls_by_func[nid] = body_calls

        # Index struct definitions
        for sname in struct_defs:
            struct_def_index[sname] = rel

        file_data[rel] = {
            "funcs": funcs,
            "includes": includes,
            "struct_refs": struct_refs,
            "calls_by_func": calls_by_func,
        }

    # ── Second pass: build edges ──────────────────────────────────

    for rel, data in file_data.items():
        # Include edges
        for inc_path in data["includes"]:
            target = _resolve_include(rel, inc_path, header_index)
            if target and target in graph.nodes:
                graph.add_edge(rel, target, "include", confidence=1.0)

        # Call edges
        for caller_id, callees in data["calls_by_func"].items():
            for callee_name in callees:
                targets = func_index.get(callee_name, [])
                if len(targets) == 1:
                    graph.add_edge(caller_id, targets[0], "call", confidence=0.95)
                elif len(targets) > 1:
                    # Ambiguous: distribute confidence
                    conf = 0.95 / len(targets)
                    for t in targets:
                        graph.add_edge(caller_id, t, "call", confidence=round(conf, 3))

        # Type/struct usage edges
        for sname in data["struct_refs"]:
            def_file = struct_def_index.get(sname)
            if def_file and def_file != rel:
                graph.add_edge(rel, def_file, "type_use", confidence=0.7)

        # Test edges
        if _is_test_file(rel):
            # Collect all source files called from this test file
            called_files = set()
            for caller_id, callees in data["calls_by_func"].items():
                for callee_name in callees:
                    for target_id in func_index.get(callee_name, []):
                        target_node = graph.nodes[target_id]
                        target_file = target_node["path"]
                        if not _is_test_file(target_file):
                            called_files.add(target_file)
            for src_file in called_files:
                graph.add_edge(rel, src_file, "test", confidence=0.8)

    return graph


def _resolve_include(
    source_file: str, include_path: str, header_index: dict[str, str]
) -> str | None:
    """Resolve a quoted #include to a file node id."""
    # Try relative path first
    source_dir = str(Path(source_file).parent)
    candidate = (Path(source_dir) / include_path).as_posix()
    # Normalize
    candidate = str(Path(candidate)).replace("\\", "/")
    if candidate in header_index.values():
        return candidate

    # Fallback: basename match
    basename = Path(include_path).name
    return header_index.get(basename)
