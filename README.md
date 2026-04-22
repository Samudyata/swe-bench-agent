# SWE-bench-C Multi-Agent System

An evidence-routed multi-agent pipeline for automatically resolving C-language issues from the [SWE-bench-C](https://huggingface.co/datasets/xingyaoww/swe-bench-c) benchmark.

---

## Architecture Overview

```
Issue Text
    │
    ▼
┌─────────┐       PlannerOutput
│ Planner │ ──────────────────────────────────────────────────────────────────┐
└─────────┘                                                                   │
    │                                                                         │
    │ keywords, suspected_modules, priority_functions                         │
    ▼                                                                         │
┌───────────┐      ContextBundle                                              │
│ Localizer │ ─────────────────────────────────────────────────────────────┐ │
└───────────┘                                                               │ │
    │ ranked files, snippets, confidence                                    │ │
    ▼                                                                       │ │
┌──────────────┐   FixPlan                                                  │ │
│ Diagnostician│ ──────────────────────────────────────────────────────┐   │ │
└──────────────┘                                                        │   │ │
    │ root cause, affected regions, fix description                     │   │ │
    ▼                                                                   │   │ │
┌────────┐         PatchOutput                                          │   │ │
│ Patcher│ ─────────────────────────────────────────────────────────┐  │   │ │
└────────┘                                                           │  │   │ │
    │ unified diff (git-apply ready)                                 │  │   │ │
    ▼                                                                │  │   │ │
┌──────────┐      ValidationResult                                   │  │   │ │
│ Validator│ ── apply → compile → test ──────► SUCCESS              │  │   │ │
└──────────┘                    │                                    │  │   │ │
                        FAILURE │  FeedbackMessage                   │  │   │ │
                                ├── apply_failed   ──────────────────┘  │   │ │
                                ├── compile_failed ──────────────────────┘   │ │
                                ├── test_failed    ──────────────────────────┘ │
                                ├── regression     ──────────────────────────────┘ (Diagnostician)
                                └── low_confidence ──────────────────────────────── Planner (replan)
```

**Each failure is routed back to the single agent that can fix it** — no re-running the whole pipeline from scratch on every retry.

---

## Project Status

### ✅ Person 1 — Graph Infrastructure (COMPLETE)

> **Owner**: Person 1
> **Tests**: 45/45 smoke tests passing (`scripts/test_graph.py`)

All graph infrastructure is built and ready for the other agents to consume.

| File | Description |
|---|---|
| `graph/model.py` | `DepGraph` data structure — nodes (file, function), edges (include, call, test), adjacency index |
| `graph/builder.py` | tree-sitter C parser — extracts file nodes, function nodes, call edges, include edges, test links |
| `graph/query.py` | Traversal API: `get_neighbors()`, `get_callers()`, `get_callees()`, `get_file_functions()`, `expand_hop()`, `get_test_files()` |
| `graph/scoring.py` | `keyword_search()` (name/path weighted match), `evidence_subgraph()` (BFS from keyword seeds), `extract_keywords()` |
| `graph/repo_checkout.py` | `ensure_repo_at_commit()` — clones and checks out a repo at its SWE-bench base commit |
| `graph/__init__.py` | Public API exports |
| `scripts/build_all_graphs.py` | Batch graph builder for all 179 SWE-bench-C instances |

**⚠️ Action Required**: The `graphs/` directory is currently empty. Run `scripts/build_all_graphs.py` on a machine with the target repos cloned to populate it before running the pipeline end-to-end.

---

### ✅ Person 2 — Planner + Pipeline Orchestration (COMPLETE)

> **Owner**: Person 2
> **Tests**: 38/38 passing (`scripts/test_person2.py`)

The pipeline's shared communication contract, logging backbone, LLM planner, and orchestration controller are all implemented.

#### Inter-Agent Schema (`pipeline/schema.py`)
All typed messages that flow between agents — **the source of truth for Person 3/4/5**:

| Dataclass | Direction | Fields |
|---|---|---|
| `SWEInstance` | Dataset → all agents | `instance_id`, `repo`, `base_commit`, `problem_statement`, `fail_to_pass`, `pass_to_pass` |
| `PlannerOutput` | Planner → Localizer | `keywords`, `search_hints`, `suspected_modules`, `priority_functions`, `issue_type` |
| `ContextBundle` | Localizer → Diagnostician | `candidates`, `confidence`, `file_contents`, `relevant_snippets`, `test_files` |
| `FixPlan` | Diagnostician → Patcher | `root_cause`, `affected_files`, `affected_regions`, `test_constraints`, `fix_description` |
| `PatchOutput` | Patcher → Validator | `unified_diff` (git-apply ready), `affected_files` |
| `ValidationResult` | Validator → Controller | `status`, `resolved`, `apply_ok`, `compile_ok`, `tests_passed`, `tests_failed`, `error_output` |
| `FeedbackMessage` | Controller → upstream agent | `failure_type`, `route_to`, `evidence`, `retry_number` |
| `FailureType` | Enum | `SUCCESS`, `APPLY`, `COMPILE`, `TEST`, `REGRESSION`, `LOW_CONF` |

#### Structured Logger (`pipeline/logger.py`)
- JSONL file per instance: `logs/<instance_id>_<timestamp>.jsonl`
- Compact stdout echo: `[instance_id] [agent] event  key=val`
- `PipelineLogger.load(path)` — static method to replay a run's events

#### Planner Agent (`agents/planner.py`)
- **LLM backend**: Google Gemini (via `google-genai` SDK)
- Produces `PlannerOutput` with keywords, priority functions, suspected modules
- **Two prompts**: initial plan (`plan()`) and low-confidence replan (`replan()`)
- JSON output mode with parse-failure retry (up to `max_llm_retries`)
- Migrated to `google.genai` client (not deprecated `google.generativeai`)

#### Pipeline Controller (`pipeline/controller.py`)
- Sequences all 5 agents: Planner → Localizer → Diagnostician → Patcher → Validator
- **Evidence routing table**: maps each `FailureType` to the specific agent that retries
- Per-agent retry cap (`max_retries`, default 2)
- Persists final `ValidationResult` to `results/<instance_id>.json`
- Handles `LOW_CONF` → replan without re-running all downstream agents

#### Agent Stubs (`agents/stubs.py`)
Drop-in stubs for Persons 3/4/5 that implement the required method signatures. Used for end-to-end pipeline smoke testing before teammate code is ready.

#### Config & CLI
| File | Purpose |
|---|---|
| `config.py` | `Config.from_env()` — loads all settings from `.env` |
| `run_pipeline.py` | CLI entry point with `--instance_id`, `--repo`, `--all`, `--planner-only`, `--stub-agents` |
| `.env.example` | Template for API key setup |

---

### ✅ Person 3 — Localizer + Retrieval Engine (COMPLETE)

> **Owner**: Person 3
> **Tests**: 50/50 passing (`scripts/test_person3.py`)

The Localizer is a three-pass retrieval engine — primarily deterministic (graph + grep), with optional LLM re-ranking for ablation experiments.

#### Three-Pass Retrieval Architecture

```
Pass 1: Graph keyword search
    keyword_search(graph, keywords)  →  node scores → aggregated by file

Pass 2: Grep pass
    grep_repo(repo_root, keywords + priority_functions)  →  hit density per file

Pass 3: Graph neighbourhood expansion
    evidence_subgraph(graph, keywords, top_k=5, hops=2)
    → BFS from top seeds → new files discovered via structural links

→ Composite score = graph × 0.60 + grep × 0.30 + suspected_bonus × 0.10
→ Read file contents + extract function snippets
→ Confidence scoring → ContextBundle
```

#### Confidence Score (4-component)
| Component | Weight | Meaning |
|---|---|---|
| `top_score_norm` | 0.40 | How strong is the top file's raw score? |
| `score_gap` | 0.25 | How decisively does #1 beat #2? |
| `coverage` | 0.15 | How many files did we find (vs desired 8)? |
| `module_hit_rate` | 0.20 | Did Planner's suspected files appear in results? |

If `confidence < threshold` (default 0.4), the Controller routes `LOW_CONF` back to the Planner for replanning.

#### Files Implemented

| File | Description |
|---|---|
| `agents/localizer.py` | Main `LocalizerAgent` — `localize()`, `localize_with_feedback()`, confidence scoring, optional LLM re-ranking |
| `agents/tools/grep_tool.py` | Whole-word case-insensitive regex grep over `.c`/`.h` files; skips vendor dirs; returns `GrepHit` dataclasses |
| `agents/tools/file_reader.py` | `read_file()`, `extract_snippet()` (with ±3 line context padding, 150-line cap), `extract_function_snippets()` using graph line metadata |
| `agents/tools/graph_tools.py` | `build_scored_candidates()`, `select_top_function_nodes()` — bridges Person 1's graph API to scored candidate format |
| `agents/tools/__init__.py` | Package exports |

#### Feedback / Retry Mode
When `apply_failed` is routed back, `localize_with_feedback()`:
1. Parses the rejected file path from the Validator's error evidence (e.g., `error: patch failed: src/jv.c:145`)
2. Expands the graph 1 hop from that seed (`min_confidence=0.7`)
3. Re-runs the full three-pass retrieval with the expanded neighbourhood

---

### ✅ Person 4 — Diagnostician + Patcher (COMPLETE)

> **Owner**: Person 4
> **Tests**: `scripts/test_person4.py`

Both agents use the `google-genai` SDK. By default they authenticate with
`GEMINI_API_KEY` (matching the Planner); if `VERTEXAI_PROJECT` is set they
transparently switch to Vertex AI.

| File | Role |
|---|---|
| `agents/diagnostician.py` | `DiagnosticianAgent.diagnose` / `.revise` — produces `FixPlan` with root cause, affected regions, and test constraints. |
| `agents/patcher.py` | `PatcherAgent.patch` / `.patch_with_feedback` — emits a `git apply`-compatible unified diff. Context lines are copied from real file content in `ContextBundle.file_contents` (falls back to disk via `repo_root`). |

---

### ✅ Person 5 — Validator (COMPLETE)

> **Owner**: Person 5
> **Tests**: `scripts/test_person5.py`

| File | Role |
|---|---|
| `agents/validator.py` | `ValidatorAgent.validate` — runs `git apply` → `make` → FAIL_TO_PASS tests → PASS_TO_PASS regression check. Stops at first failure and returns the appropriate `FailureType` (`APPLY` / `COMPILE` / `TEST` / `REGRESSION`) so the controller can route evidence back to the responsible agent. |

`repo_root` is set per-instance by `run_pipeline.py` via
`ValidatorAgent.set_repo_root(...)`. Compile / test timeouts are driven by
`COMPILE_TIMEOUT` and `TEST_TIMEOUT` env vars (see `config.py`).

---

## Test Results Summary

| Suite | Tests | Status |
|---|---|---|
| `scripts/test_graph.py` | 45 | ✅ All passing |
| `scripts/test_person2.py` | 38 | ✅ All passing (live LLM: SKIP — quota, key is valid) |
| `scripts/test_person3.py` | 50 | ✅ All passing (real-repo: SKIP — repos/ not cloned yet) |
| `scripts/test_person4.py` | Diagnostician + Patcher | ✅ integration tests written |
| `scripts/test_person5.py` | Validator | ✅ integration tests written |

---

## Quick Start

### 1. Install dependencies

```bash
pip install tree-sitter tree-sitter-c google-genai python-dotenv
# Optional for HuggingFace dataset auto-download:
pip install datasets
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Build graphs (Person 1 prerequisite)

```bash
# Clone repos first (example for jq):
git clone https://github.com/jqlang/jq repos/jqlang__jq
cd repos/jqlang__jq && git checkout <base_commit> && cd ../..

# Build the graph for all instances:
python scripts/build_all_graphs.py
```

### 4. Run tests

```bash
# No API key needed for Person 1 and 3 tests:
python scripts/test_graph.py
python -X utf8 scripts/test_person3.py

# Needs GEMINI_API_KEY set in .env:
python -X utf8 scripts/test_person2.py
```

### 5. Run the pipeline

```bash
# Smoke test with stubs (no repos or advanced agents needed):
python run_pipeline.py --instance_id jq-493__jqlang__jq --stub-agents

# Planner only (useful for testing Planner + Localizer together):
python run_pipeline.py --instance_id jq-493__jqlang__jq --planner-only

# Full pipeline:
python run_pipeline.py --instance_id jq-493__jqlang__jq

# Run all jq instances:
python run_pipeline.py --repo jqlang/jq

# Run all 179 SWE-bench-C instances:
python run_pipeline.py --all
```

---

## File Tree

```
swe-bench-agent/
│
├── config.py                    # Environment-driven configuration
├── run_pipeline.py              # CLI entry point
├── .env.example                 # API key template
├── .gitignore
│
├── graph/                       # ✅ Person 1 — Graph Infrastructure
│   ├── __init__.py
│   ├── model.py                 # DepGraph data structure
│   ├── builder.py               # tree-sitter C parser & graph builder
│   ├── query.py                 # Traversal API (expand_hop, get_callers, ...)
│   ├── scoring.py               # keyword_search, evidence_subgraph
│   └── repo_checkout.py         # ensure_repo_at_commit
│
├── pipeline/                    # ✅ Person 2 — Schema, Logger, Controller
│   ├── __init__.py
│   ├── schema.py                # ALL inter-agent typed messages
│   ├── logger.py                # JSONL structured event logger
│   └── controller.py            # Orchestrator + evidence routing
│
├── agents/                      # Agents
│   ├── __init__.py
│   ├── planner.py               # ✅ Person 2 — Gemini LLM Planner
│   ├── localizer.py             # ✅ Person 3 — Three-pass retrieval engine
│   ├── stubs.py                 # Stubs for smoke-testing (run_pipeline.py --stub-agents)
│   ├── diagnostician.py         # ✅ Person 4 — DiagnosticianAgent
│   ├── patcher.py               # ✅ Person 4 — PatcherAgent
│   ├── validator.py             # ✅ Person 5 — ValidatorAgent
│   └── tools/                   # ✅ Person 3 — Retrieval tools
│       ├── __init__.py
│       ├── grep_tool.py         # Regex grep over repo source files
│       ├── file_reader.py       # File content + function snippet extraction
│       └── graph_tools.py       # Scored file candidate builder
│
├── scripts/
│   ├── build_all_graphs.py      # Build & cache graphs for all instances
│   ├── test_graph.py            # ✅ Person 1 smoke tests (45 tests)
│   ├── test_person2.py          # ✅ Person 2 tests (38 tests)
│   └── test_person3.py          # ✅ Person 3 tests (50 tests)
│
├── graphs/                      # ⚠️  EMPTY — run build_all_graphs.py
├── repos/                       # ⚠️  EMPTY — clone target repos here
└── results/                     # Pipeline output (auto-created)
```

---

## Integration Guide for Persons 4 & 5

### What you receive (ContextBundle from Localizer)

```python
bundle = ContextBundle(
    instance_id = "jq-493__jqlang__jq",
    candidates  = [
        LocalizerCandidate(
            file_path  = "src/jv.c",
            score      = 0.87,            # composite relevance score
            reason     = "graph_score=4.2; grep_hits=12; suspected_by_planner",
            functions  = ["src/jv.c::jv_parse", "src/jv.c::jv_load"],
        ),
        ...                               # up to 8 candidates, ranked
    ],
    confidence      = 0.74,              # > 0.4 means Planner agrees
    file_contents   = {
        "src/jv.c": "<full source text of the file>",
        ...
    },
    relevant_snippets = {
        "src/jv.c::jv_parse": "// --- 142-158 ---\nint jv_parse(...) { ... }",
        ...
    },
    test_files = ["tests/jq.test", "tests/jq_test.c"],
)
```

### What you must produce (FixPlan → PatchOutput → ValidationResult)

See `pipeline/schema.py` for the full field definitions.  The stubs in
`agents/stubs.py` show the exact method signatures your classes must implement.

### Running integration tests while implementing

```bash
# Test with real Localizer, stub Patcher/Validator:
python run_pipeline.py --instance_id jq-493__jqlang__jq --stub-agents
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *required* | Google Gemini API key ([get one here](https://aistudio.google.com/apikey)) |
| `MODEL_NAME` | `gemini-1.5-flash` | Gemini model for Planner (and optional Localizer re-ranking) |
| `CONF_THRESHOLD` | `0.4` | Localizer confidence below this → replan |
| `MAX_RETRIES` | `2` | Max retries per agent slot on failure |
| `LOG_DIR` | `logs/` | JSONL log output directory |
| `GRAPHS_DIR` | `graphs/` | Pre-built graph JSON files |
| `REPOS_DIR` | `repos/` | Cloned repository checkouts |
| `RESULTS_DIR` | `results/` | Final ValidationResult JSON files |
