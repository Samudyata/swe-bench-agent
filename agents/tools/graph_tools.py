"""Graph-based scored candidate builder for the Localizer agent.

Takes raw node scores from ``graph.scoring.keyword_search`` and aggregates
them to the *file* level, merging with grep hit density and a Planner-provided
suspected-module bonus into a single composite score.

Score formula
-------------
    composite(file) =
        graph_score(file) * GRAPH_WEIGHT      (0.60)
      + grep_density(file) * GREP_WEIGHT      (0.30)
      + suspected_bonus(file) * BONUS_WEIGHT  (0.10)

    where:
      graph_score  = Σ keyword_search_score(node)  for all nodes in the file,
                     normalised to [0, 1] by the max observed across all files
      grep_density = hit_count_for_file / max_hit_count_across_files  ∈ [0, 1]
      suspected_bonus = SUSPECTED_BONUS if file in plan.suspected_modules else 0

Weights rationale
-----------------
Graph signal dominates (0.60): call chains and include edges capture
cross-file relationships invisible to grep.  Grep (0.30) catches literals in
comments, macros, and string tables.  The suspected-module bonus (0.10) gives a
small nudge when the Planner and retrieval agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from graph.model import DepGraph
from graph.scoring import keyword_search
from graph.query import get_file_functions

# Composite score weights
GRAPH_WEIGHT: float = 0.60
GREP_WEIGHT: float = 0.30
BONUS_WEIGHT: float = 0.10

# Bonus added when the file appears in Planner's suspected_modules
SUSPECTED_BONUS: float = 1.0   # raw bonus; normalisation then puts it in [0, 0.1]


@dataclass
class ScoredFile:
    """A source file with its composite relevance score and contributing nodes."""
    file_path: str                        # repo-relative POSIX path
    graph_score: float                    # raw (unnormalised) graph score
    grep_score: float                     # normalised grep density [0, 1]
    composite_score: float                # final weighted combination [0, 1+]
    contributing_nodes: list[str] = field(default_factory=list)  # func node IDs
    is_suspected: bool = False            # True if in Planner's suspected_modules
    grep_hit_count: int = 0               # raw number of grep hits in this file

    def __repr__(self) -> str:
        return (
            f"ScoredFile({self.file_path!r}, "
            f"composite={self.composite_score:.3f}, "
            f"graph={self.graph_score:.2f}, "
            f"grep={self.grep_score:.2f}, "
            f"suspected={self.is_suspected})"
        )


def build_scored_candidates(
    graph: DepGraph,
    keywords: list[str],
    grep_hits: dict[str, list],          # {file_path: [GrepHit, ...]}
    suspected_modules: list[str],
    top_k: int = 10,
) -> list[ScoredFile]:
    """Build a ranked list of ScoredFile objects.

    Algorithm
    ---------
    1. Run ``keyword_search(graph, keywords)`` → ``(node_id, score)`` pairs.
    2. Aggregate by file: sum scores of all nodes belonging to each file.
    3. Also collect the node IDs that contributed (for snippet extraction later).
    4. Normalise graph scores to [0, 1].
    5. Compute grep density per file: ``hits / max_hits``.
    6. Compute composite score using the weights above.
    7. Apply suspected-module bonus.
    8. Sort descending, return top ``top_k``.

    Args:
        graph:             Pre-built DepGraph for the repository.
        keywords:          Search keywords from PlannerOutput.
        grep_hits:         Output of ``grep_repo()`` — ``{file: [GrepHit]}``.
        suspected_modules: Files the Planner thinks are relevant.
        top_k:             Maximum number of candidates to return.

    Returns:
        Ranked list of :class:`ScoredFile`, highest composite score first.
    """
    # ── Step 1–3: graph scores aggregated by file ─────────────────────────────
    raw_graph: dict[str, float] = defaultdict(float)         # file → score
    contributing: dict[str, list[str]] = defaultdict(list)   # file → [node_ids]

    if keywords:
        for node_id, score in keyword_search(graph, keywords):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            file_path = node["path"]
            raw_graph[file_path] += score
            contributing[file_path].append(node_id)

    # ── Step 4: normalise graph scores ────────────────────────────────────────
    max_graph = max(raw_graph.values(), default=1.0) or 1.0
    norm_graph: dict[str, float] = {
        fp: s / max_graph for fp, s in raw_graph.items()
    }

    # ── Step 5: grep density ──────────────────────────────────────────────────
    grep_counts: dict[str, int] = {fp: len(hits) for fp, hits in grep_hits.items()}
    max_grep = max(grep_counts.values(), default=1) or 1
    norm_grep: dict[str, float] = {
        fp: c / max_grep for fp, c in grep_counts.items()
    }

    # ── Step 6–7: union of all candidate files ────────────────────────────────
    suspected_set = set(suspected_modules)
    all_files: set[str] = set(raw_graph) | set(grep_hits) | suspected_set

    scored: list[ScoredFile] = []
    for fp in all_files:
        gs = norm_graph.get(fp, 0.0)
        gp = norm_grep.get(fp, 0.0)
        is_susp = fp in suspected_set
        bonus = SUSPECTED_BONUS if is_susp else 0.0

        composite = (
            gs * GRAPH_WEIGHT
            + gp * GREP_WEIGHT
            + bonus * BONUS_WEIGHT
        )

        scored.append(ScoredFile(
            file_path=fp,
            graph_score=raw_graph.get(fp, 0.0),
            grep_score=gp,
            composite_score=round(composite, 4),
            contributing_nodes=contributing.get(fp, []),
            is_suspected=is_susp,
            grep_hit_count=grep_counts.get(fp, 0),
        ))

    # ── Step 8: sort and truncate ─────────────────────────────────────────────
    scored.sort(key=lambda s: s.composite_score, reverse=True)
    return scored[:top_k]


def select_top_function_nodes(
    graph: DepGraph,
    scored_files: list[ScoredFile],
    keywords: list[str],
    max_funcs_per_file: int = 5,
) -> dict[str, list[str]]:
    """For each top file, select the most relevant function node IDs.

    Used by the Localizer to decide which snippets to extract.

    Args:
        graph:              The dependency graph.
        scored_files:       Output of ``build_scored_candidates``.
        keywords:           Keywords to score functions within each file.
        max_funcs_per_file: Cap on how many functions to extract per file.

    Returns:
        ``{file_path: [func_node_id, ...]}`` — functions ranked by score within
        each file, capped at ``max_funcs_per_file``.
    """
    result: dict[str, list[str]] = {}

    # Pre-compute keyword-level node scores for filtering
    node_scores: dict[str, float] = {}
    if keywords:
        for nid, score in keyword_search(graph, keywords):
            node_scores[nid] = score

    for sf in scored_files:
        # Get all functions in this file
        funcs = get_file_functions(graph, sf.file_path)

        # Sort by keyword score (higher = more relevant), then by start line
        def _rank(fn: dict) -> tuple:
            nid = fn["id"]
            return (-node_scores.get(nid, 0.0), fn.get("start_line", 0))

        funcs_sorted = sorted(funcs, key=_rank)
        result[sf.file_path] = [f["id"] for f in funcs_sorted[:max_funcs_per_file]]

    return result
