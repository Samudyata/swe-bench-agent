# swe-bench-agent

Evidence-routed multi-agent system for [SWE-bench](https://www.swebench.com/) and [SWE-bench-C](https://github.com/SWE-bench-c/SWE-bench-c).

## Overview

An automated software engineering system that resolves real-world GitHub issues by generating patches for C and Python repositories. Instead of single-shot patch generation (which achieves 0% resolution on SWE-bench-C), we use an **evidence-routed multi-agent architecture** where failures produce structured evidence that gets routed to the specific agent capable of addressing them.

### Why "Evidence-Routed"?

Standard multi-agent systems retry the entire pipeline on failure. Our system classifies each failure (patch doesn't apply? compilation error? test failure?) and routes it back to the specific agent that can fix it — with the error message as actionable evidence.

| Failure Type | Evidence | Routed To | Action |
|---|---|---|---|
| `git apply` fails | Rejected hunk + file context | Localizer | Re-search with expanded graph neighborhood |
| Compilation error | `gcc` error message + line number | Patcher | Retry with compiler error appended |
| Test failure | Test output + expected vs actual | Diagnostician | Revise root cause hypothesis |
| Low confidence localization | Retrieval scores below threshold | Planner | Refine search strategy |

## Architecture

```
OFFLINE (once per repo)
  Repository @ base_commit --> tree-sitter parse --> Code Dependency Graph

ONLINE (per issue)
  Issue Text
    |
    v
  [Planner] --> classify issue, generate search strategy
    |
    v
  [Localizer] --> grep + graph traversal --> ranked source files
    |
    v
  [Diagnostician] --> read code + run tests --> root cause + fix plan
    |
    v
  [Patcher] --> generate unified diff from real source code
    |
    v
  [Validator] --> git apply --> compile --> run tests
    |                                          |
    +--- on failure: route evidence back ------+
         to the appropriate agent (max 2 retries)
```

### The Four Agents

- **Planner** — Analyzes issue text, classifies it (bug/feature/refactor), and produces a search strategy with priority keywords. Pure reasoning, no tools.
- **Localizer** — Finds relevant files and functions using grep, file reading, and graph neighbor queries. Operates iteratively with confidence-based expansion.
- **Diagnostician** — Reads localized code and test files, identifies root cause, produces a structured fix plan with affected lines and constraints.
- **Patcher** — Generates a syntactically correct unified diff using real source code. Context lines are copied from actual files, not hallucinated.

### Code Dependency Graph

The key infrastructure enabling cross-file reasoning. Built offline per repository using [tree-sitter](https://tree-sitter.github.io/) (no compilation required).

**Why not just grep?** A fix in `compress.c` may require a struct from `zstd_internal.h` and a function from `compress_impl.c` — neither contains issue keywords. The graph captures these structural relationships explicitly.

**Smarter-than-static mechanisms:**
1. **Edge confidence scoring** — Every edge gets a confidence (0.0-1.0) based on resolution certainty. Ambiguous call targets get proportionally lower confidence.
2. **Keyword-weighted traversal** — Issue keywords score nodes, then BFS expands only through edges above a confidence threshold, producing focused "evidence subgraphs" instead of noisy full neighborhoods.

#### Graph Schema

Each graph is serialized as a JSON file with the following top-level structure:

```json
{
  "repo":   "<owner/name>",
  "commit": "<git-commit-hash>",
  "nodes":  { "<node_id>": { ... } },
  "edges":  [ { ... } ]
}
```

**Node types:**

There are two kinds of nodes, distinguished by the `kind` field.

*File node* — represents a `.c` or `.h` source file:

```json
"src/jv.c": {
  "id":   "src/jv.c",
  "kind": "file",
  "path": "src/jv.c"
}
```

Node ID format: `<rel/path/to/file.c>` (POSIX separators, relative to repo root).

*Function node* — represents a C function definition:

```json
"src/jv.c::jv_copy": {
  "id":         "src/jv.c::jv_copy",
  "kind":       "function",
  "path":       "src/jv.c",
  "name":       "jv_copy",
  "start_line": 42,
  "end_line":   58
}
```

Node ID format: `<path>::<name>`. If two functions in the same file share a name, the second gets a `@<start_line>` suffix: `<path>::<name>@<start_line>`.

**Edge schema:**

```json
{
  "source":     "<node_id>",
  "target":     "<node_id>",
  "kind":       "call",
  "weight":     1.0,
  "confidence": 0.95
}
```

| Field | Type | Description |
|---|---|---|
| `source` | string | Node ID of the source |
| `target` | string | Node ID of the target |
| `kind` | string | Edge type (see below) |
| `weight` | float | Always `1.0` (reserved) |
| `confidence` | float | Resolution confidence in `[0.0, 1.0]` |

**Edge types and confidence values:**

| `kind` | Source | Target | What It Captures | Confidence |
|---|---|---|---|---|
| `include` | file | file | File A `#include`s File B (local includes only) | `1.0` |
| `call` | function | function | Function F calls Function G (unique resolution) | `0.95` |
| `call` | function | function | Function F calls G (ambiguous, N candidates) | `0.95/N` |
| `type_use` | file | file | File A references a struct defined in File B | `0.7` |
| `test` | file | file | Test file T exercises source file S | `0.8` |

**Builder notes:**
- Skipped directories: `vendor`, `deps`, `contrib`, `third_party`, `third-party`, `external`, `node_modules`, `.git`, `build`, `cmake-build`
- System headers (`<stdio.h>` etc.) are excluded; only quoted `#include "..."` directives produce edges
- Test file detection: path matches `tests?/`, `test_`, or `_test.c` (case-insensitive)
- `type_use` edges are file-level (not function-level), as struct resolution operates at the translation-unit level
- The adjacency indexes (`adj`, `rev`) are derived from `edges` at load time and are not stored in JSON

#### Test Results

The graph infrastructure passes **45/45** smoke tests covering:

| Category | Checks | What's Validated |
|---|---|---|
| Build | 5 | File/function node counts, all 4 edge types present |
| Node structure | 8 | Correct fields (`id`, `kind`, `path`, `name`, `start_line`, `end_line`) |
| Edge structure | 6 | Source/target point to correct node types, confidence in [0, 1] |
| Confidence scoring | 3 | Include=1.0, call<=0.95, test=0.8 |
| Query API | 11 | `get_callers`, `get_callees`, `expand_hop`, `get_test_files`, `get_neighbors`, confidence filtering |
| Scoring | 5 | Keyword extraction, search ranking, exact-match bonus, evidence subgraph |
| Serialization | 5 | JSON round-trip preserves nodes, edges, metadata, adjacency indexes |

**Tested on `jqlang/jq`:**
```
47 files, 510 functions, 3,896 edges (87 include, 3,785 call, 10 type_use, 14 test)
Built in 1.4 seconds | Serializes to 792 KB JSON
```

Run tests with:
```bash
python scripts/test_graph.py
```

Run per-repo sanity-check stats with:
```bash
python scripts/stats_per_repo.py
```

## Project Structure

```
swe-bench-agent/
  graph/
    __init__.py          # Public API exports
    model.py             # DepGraph class, JSON serialization
    builder.py           # tree-sitter C parser, edge extraction (~150 LOC)
    query.py             # Query API: callers, callees, expand_hop, test files
    scoring.py           # Keyword search, evidence subgraph extraction
    repo_checkout.py     # Git clone/checkout helper
  scripts/
    build_all_graphs.py  # Build graphs for all 179 SWE-bench-C instances
    test_graph.py        # Smoke tests (45 checks)
    stats_per_repo.py    # Per-repo sanity-check stats
  graphs/                # Serialized JSON graphs (gitignored)
  repos/                 # Cloned repositories (gitignored)
```

## Setup

```bash
pip install tree-sitter tree-sitter-c
```

Build graphs for all SWE-bench-C instances:
```bash
python scripts/build_all_graphs.py
```

## Benchmarks

**SWE-bench-C** — 179 instances from 3 C repositories:
- `facebook/zstd` (44 instances) — lossless compression library
- `jqlang/jq` (45 instances) — command-line JSON processor
- `redis/redis` (90 instances) — in-memory key-value store

**SWE-bench** — 300 instances from Python repositories

## Phase 1 Baseline (Motivation)

Single-shot inference without repository access achieved **0% resolution** on SWE-bench-C:

| Stage | Best Result | Bottleneck |
|---|---|---|
| Localization (`git apply`) | 9.5% (17/179) | 90%+ patches rejected — model can't reconstruct file content it's never seen |
| Compilation | 29.4% (5/17) | Missing headers, type mismatches, undeclared variables |
| Resolution | 0% (0/179) | No correct fixes produced |

Key finding: providing file paths (V2) did **not** improve localization over blind prompting (V1). The bottleneck is missing *content*, not missing *addresses*.

## Team

Gursparsh Singh Sodhi, Hithaishi Surendra, Jahnvi Seth, Samhitha Harish, Samudyata Sudarshan Jagirdar

## References

- [SWE-bench](https://arxiv.org/abs/2310.06770) — Jimenez et al., 2023
- [SWE-bench-C](https://github.com/SWE-bench-c/SWE-bench-c) — Vayavya Labs, 2025
- [tree-sitter](https://tree-sitter.github.io/) — Incremental parsing system
