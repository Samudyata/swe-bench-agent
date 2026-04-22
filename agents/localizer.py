"""Localizer agent — retrieves relevant source files and function snippets.

The Localizer is the retrieval backbone of the pipeline.  It takes the
Planner's structured search plan and the pre-built code dependency graph
and produces a ContextBundle containing ranked candidate files, full source
text, relevant function snippets, and a confidence score.

Architecture — Three-Pass Retrieval
------------------------------------

    Pass 1: Graph keyword search
        keyword_search(graph, plan.keywords)
        → scores graph nodes by name/path match
        → aggregated to file level by graph_tools.build_scored_candidates

    Pass 2: Grep pass
        grep_repo(repo_root, plan.keywords + plan.priority_functions)
        → hits in actual file content (catches comments, macros, strings)
        → merged with graph scores via composite formula

    Pass 3: Graph neighbourhood expansion
        evidence_subgraph(graph, keywords, top_k=5, hops=2)
        → BFS from top keyword seeds → discovers structurally related files
        → any new files added to candidates (graph score only)

    → Merge, sort by composite score, take top MAX_CANDIDATES files
    → Read file contents + extract function snippets (file_reader)
    → Gather test files via get_test_files(graph, ...)
    → Compute confidence score
    → Return ContextBundle

Retry / Feedback
----------------
When the Validator routes an ``apply_failed`` FeedbackMessage back, the
controller calls ``localize_with_feedback()``.  This method:
  1. Parses the rejected file path from the error evidence
  2. Expands the graph 1 hop from that seed with min_confidence=0.7
  3. Re-runs the full three-pass retrieval with the augmented seed set

Usage
-----
    from agents.localizer import LocalizerAgent
    agent = LocalizerAgent(repos_dir=Path("repos"))
    bundle = agent.localize(instance, plan, graph)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from graph.model import DepGraph
from graph.query import expand_hop, get_test_files
from graph.scoring import evidence_subgraph
from graph.repo_checkout import ensure_repo_at_commit

from pipeline.schema import (
    ContextBundle,
    FeedbackMessage,
    LocalizerCandidate,
    PlannerOutput,
    SWEInstance,
)
from agents.tools.grep_tool import grep_repo, hit_density
from agents.tools.file_reader import extract_function_snippets, read_top_files
from agents.tools.graph_tools import (
    ScoredFile,
    build_scored_candidates,
    select_top_function_nodes,
)

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

# Maximum number of candidate files returned
MAX_CANDIDATES: int = 8

# Maximum functions to extract snippets for per file
MAX_FUNCS_PER_FILE: int = 5

# Grep: hard-stop after this many hits per keyword (avoids flooding on generic terms)
MAX_GREP_HITS_PER_KW: int = 60

# Graph expansion: min edge confidence for BFS during feedback mode
FEEDBACK_MIN_CONFIDENCE: float = 0.7

# Normal graph expansion confidence threshold
NORMAL_MIN_CONFIDENCE: float = 0.3


# ── Confidence scoring weights ────────────────────────────────────────────────

_W_TOP_SCORE  = 0.40   # how strong is the top file's score?
_W_SCORE_GAP  = 0.25   # how much does #1 beat #2? (certainty)
_W_COVERAGE   = 0.15   # did we find enough candidates?
_W_MODULE_HIT = 0.20   # did Planner's suspected files appear in results?


class LocalizerAgent:
    """Evidence-based source file retrieval agent.

    Args:
        repos_dir:          Directory containing cloned repositories.
                            Repo for ``jqlang/jq`` lives at ``repos/jqlang__jq/``.
        max_candidates:     Maximum files in the returned ContextBundle.
        max_funcs_per_file: Max function snippets extracted per file.
        auto_checkout:      If True, call ``ensure_repo_at_commit`` when the
                            repo directory does not exist (requires git on PATH).
        use_llm_rerank:     If True, call Voyager to re-rank the top candidates.
                            Off by default for the baseline ablation.
        model_name:         Model used for optional LLM re-ranking
                            (default: qwen3-30b-a3b-instruct-2507).
        api_key:            Voyager API key; falls back to OPENAI_API_KEY env.
        base_url:           Voyager base URL; falls back to OPENAI_API_BASE env.
    """

    def __init__(
        self,
        repos_dir: Path | str = Path("repos"),
        max_candidates: int = MAX_CANDIDATES,
        max_funcs_per_file: int = MAX_FUNCS_PER_FILE,
        auto_checkout: bool = True,
        use_llm_rerank: bool = False,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.repos_dir = Path(repos_dir)
        self.max_candidates = max_candidates
        self.max_funcs_per_file = max_funcs_per_file
        self.auto_checkout = auto_checkout
        self.use_llm_rerank = use_llm_rerank
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url

    # ── Public interface ──────────────────────────────────────────────────────

    def localize(
        self,
        instance: SWEInstance,
        plan: PlannerOutput,
        graph: DepGraph,
    ) -> ContextBundle:
        """Run the three-pass retrieval and return a populated ContextBundle.

        Args:
            instance: SWE-bench task (repo, commit, issue text).
            plan:     PlannerOutput with keywords and search hints.
            graph:    Pre-built dependency graph for the repo at base_commit.

        Returns:
            ContextBundle ready for the Diagnostician.
        """
        repo_root = self._resolve_repo(instance)
        keywords  = self._collect_keywords(plan)

        logger.info("[%s] Localizing with %d keywords: %s",
                    instance.instance_id, len(keywords), keywords[:6])

        scored = self._three_pass_retrieval(repo_root, graph, keywords, plan)
        return self._build_bundle(instance, scored, repo_root, graph, keywords)

    def localize_with_feedback(
        self,
        instance: SWEInstance,
        plan: PlannerOutput,
        graph: DepGraph,
        feedback: FeedbackMessage,
    ) -> ContextBundle:
        """Re-localise after an ``apply_failed`` FeedbackMessage.

        Parses the rejected file from the Validator's error evidence, expands
        the graph one hop from that seed, then re-runs the full retrieval
        with the enriched seed set.

        Args:
            instance: SWE-bench task.
            plan:     The plan used in the previous failed attempt.
            graph:    Pre-built dependency graph.
            feedback: FeedbackMessage from the Validator (apply_failed).

        Returns:
            Updated ContextBundle with expanded neighbourhood.
        """
        repo_root = self._resolve_repo(instance)
        keywords  = self._collect_keywords(plan)

        # Parse the failing file from error evidence
        extra_seeds: set[str] = self._parse_rejected_file(feedback.evidence, graph)
        logger.info("[%s] Feedback expand: extra seeds=%s",
                    instance.instance_id, extra_seeds)

        # 1-hop BFS from the rejected file to discover neighbours
        expanded: set[str] = set()
        if extra_seeds:
            expanded = expand_hop(
                graph,
                extra_seeds,
                hops=1,
                min_confidence=FEEDBACK_MIN_CONFIDENCE,
            )

        scored = self._three_pass_retrieval(
            repo_root, graph, keywords, plan, extra_file_seeds=expanded
        )
        return self._build_bundle(instance, scored, repo_root, graph, keywords)

    # ── Retrieval passes ──────────────────────────────────────────────────────

    def _three_pass_retrieval(
        self,
        repo_root: Path,
        graph: DepGraph,
        keywords: list[str],
        plan: PlannerOutput,
        extra_file_seeds: set[str] | None = None,
    ) -> list[ScoredFile]:
        """Run three complementary retrieval passes and merge results."""

        # ── Pass 1: graph keyword search + grep ───────────────────────────────
        grep_terms = list(dict.fromkeys(keywords + plan.priority_functions))
        grep_hits = grep_repo(
            repo_root,
            grep_terms,
            max_hits_per_keyword=MAX_GREP_HITS_PER_KW,
        )
        logger.info("[localizer] grep: %d files with hits", len(grep_hits))

        scored = build_scored_candidates(
            graph=graph,
            keywords=keywords,
            grep_hits=grep_hits,
            suspected_modules=plan.suspected_modules,
            top_k=self.max_candidates * 2,  # over-fetch before expansion merge
        )

        # ── Pass 2: graph neighbourhood expansion from top-5 seeds ───────────
        top_nodes: set[str] = set()
        for sf in scored[:5]:
            for nid in sf.contributing_nodes[:3]:
                top_nodes.add(nid)
            # Also seed from the file node itself
            if sf.file_path in graph.nodes:
                top_nodes.add(sf.file_path)

        # Add any extra seeds from feedback
        if extra_file_seeds:
            top_nodes |= extra_file_seeds

        if top_nodes:
            expanded_ids = evidence_subgraph(
                graph, keywords, top_k=5, hops=2,
                min_confidence=NORMAL_MIN_CONFIDENCE,
            )
            # Extract files from expanded node set
            new_files: set[str] = set()
            for nid in expanded_ids:
                node = graph.nodes.get(nid)
                if node and node["kind"] == "file":
                    new_files.add(node["path"])
                elif node and node["kind"] == "function":
                    new_files.add(node["path"])

            # Add newly discovered files not already in scored
            existing_fps = {s.file_path for s in scored}
            for fp in new_files - existing_fps:
                # Give them graph-based score only
                scored.append(ScoredFile(
                    file_path=fp,
                    graph_score=0.1,        # small default — they came via expansion
                    grep_score=hit_density(grep_hits).get(fp, 0.0),
                    composite_score=0.1 * 0.6 + hit_density(grep_hits).get(fp, 0.0) * 0.3,
                    contributing_nodes=[],
                    is_suspected=fp in set(plan.suspected_modules),
                    grep_hit_count=len(grep_hits.get(fp, [])),
                ))

        # Final sort + truncate
        scored.sort(key=lambda s: s.composite_score, reverse=True)
        top_scored = scored[:self.max_candidates]
        logger.info("[localizer] top candidates: %s",
                    [(s.file_path, round(s.composite_score, 3)) for s in top_scored])
        return top_scored

    # ── Bundle construction ───────────────────────────────────────────────────

    def _build_bundle(
        self,
        instance: SWEInstance,
        scored: list[ScoredFile],
        repo_root: Path,
        graph: DepGraph,
        keywords: list[str],
    ) -> ContextBundle:
        """Convert scored files into a ContextBundle for the Diagnostician."""

        # Select which function nodes to extract snippets for
        func_nodes_by_file = select_top_function_nodes(
            graph, scored, keywords, self.max_funcs_per_file
        )
        all_func_node_ids = [
            nid
            for nids in func_nodes_by_file.values()
            for nid in nids
        ]

        # Read full file contents for top candidates
        file_contents = read_top_files(
            repo_root, [sf.file_path for sf in scored]
        )

        # Extract function snippets
        relevant_snippets = extract_function_snippets(
            repo_root, graph, all_func_node_ids
        )

        # Collect test files for all top candidate source files
        test_file_set: set[str] = set()
        for sf in scored:
            for tf in get_test_files(graph, sf.file_path):
                test_file_set.add(tf)

        # Build LocalizerCandidate objects
        candidates = []
        for sf in scored:
            # Generate a concise human-readable reason
            reason_parts = []
            if sf.graph_score > 0:
                reason_parts.append(f"graph_score={sf.graph_score:.2f}")
            if sf.grep_hit_count > 0:
                reason_parts.append(f"grep_hits={sf.grep_hit_count}")
            if sf.is_suspected:
                reason_parts.append("suspected_by_planner")
            reason = "; ".join(reason_parts) or "expansion neighbour"

            candidates.append(LocalizerCandidate(
                file_path=sf.file_path,
                score=sf.composite_score,
                reason=reason,
                functions=func_nodes_by_file.get(sf.file_path, []),
            ))

        # Compute confidence
        confidence = self._compute_confidence(scored, keywords)
        logger.info("[%s] Localizer confidence=%.3f  candidates=%d",
                    instance.instance_id, confidence, len(candidates))

        # Optional LLM re-ranking
        if self.use_llm_rerank and candidates:
            candidates = self._llm_rerank(candidates, instance)

        return ContextBundle(
            instance_id=instance.instance_id,
            candidates=candidates,
            confidence=confidence,
            file_contents=file_contents,
            relevant_snippets=relevant_snippets,
            test_files=sorted(test_file_set),
        )

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        scored: list[ScoredFile],
        keywords: list[str],
    ) -> float:
        """Compute overall localization confidence in [0.0, 1.0].

        Components
        ----------
        top_score_norm : float
            Normalised top composite score.  If no file scores above a
            baseline it signals that keywords found nothing useful.
        score_gap : float
            Relative gap between #1 and #2.  A large gap means one file
            is clearly dominant (high certainty).  A small gap means we
            are choosing between equally likely candidates.
        coverage : float
            Fraction of the desired MAX_CANDIDATES we actually found.
            Finding only 1-2 files (vs 8) may indicate poor match.
        module_hit_rate : float
            Fraction of Planner's suspected_modules that appear in results.
            If the Planner named ``src/compress.c`` and we found it → both
            agree → higher confidence.  If suspected_modules is empty we
            use a neutral 0.5.
        """
        if not scored:
            return 0.0

        # Maximum theoretical composite score (perfect graph + grep + bonus)
        max_possible = 1.0 * 0.6 + 1.0 * 0.3 + 1.0 * 0.1   # = 1.0
        top_score_norm = min(scored[0].composite_score / max_possible, 1.0)

        # Score gap between first and second candidate
        if len(scored) >= 2 and scored[0].composite_score > 0:
            gap = (scored[0].composite_score - scored[1].composite_score)
            score_gap = min(gap / scored[0].composite_score, 1.0)
        else:
            score_gap = 1.0   # only one candidate → no ambiguity

        # Coverage
        coverage = min(len(scored) / self.max_candidates, 1.0)

        # Module hit rate
        suspected = {sf.file_path for sf in scored if sf.is_suspected}
        total_suspected = sum(1 for sf in scored if sf.is_suspected)
        # Count how many suspected modules we actually retrieved
        if any(sf.is_suspected for sf in scored[:self.max_candidates]):
            # At least some suspected files found in our candidates
            module_hit_rate = total_suspected / max(
                len([sf for sf in scored if sf.is_suspected]) +
                len([sf for sf in scored if not sf.is_suspected and sf.is_suspected]), 1
            )
            # Simpler version: fraction of top-k that were suspected
            module_hit_rate = total_suspected / max(len(scored), 1)
        else:
            # No suspected_modules set by Planner → neutral
            module_hit_rate = 0.5

        confidence = (
            top_score_norm * _W_TOP_SCORE
            + score_gap    * _W_SCORE_GAP
            + coverage     * _W_COVERAGE
            + module_hit_rate * _W_MODULE_HIT
        )
        return round(min(confidence, 1.0), 4)

    # ── Feedback helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_rejected_file(evidence: str, graph: DepGraph) -> set[str]:
        """Extract file paths from a ``git apply`` rejection message.

        Looks for patterns like:
          ``error: patch failed: src/jv.c:145``
          ``patching file src/compress.c``
          ``error: src/jq.c: does not match index``

        Returns:
            Set of repo-relative file paths that exist in the graph.
        """
        found: set[str] = set()
        # Pattern 1: "patch failed: <path>:<lineno>"
        for m in re.finditer(r"patch failed:\s*([\w./\-]+):\d+", evidence):
            fp = m.group(1)
            if fp in graph.nodes:
                found.add(fp)
        # Pattern 2: "patching file <path>"
        for m in re.finditer(r"patching file\s+([\w./\-]+)", evidence):
            fp = m.group(1)
            if fp in graph.nodes:
                found.add(fp)
        # Pattern 3: generic path-like token that exists in graph
        if not found:
            for m in re.finditer(r"([\w./\-]+\.(?:c|h))", evidence):
                fp = m.group(1)
                if fp in graph.nodes:
                    found.add(fp)
        return found

    # ── Repo resolution ───────────────────────────────────────────────────────

    def _resolve_repo(self, instance: SWEInstance) -> Path:
        """Resolve the local repo path, optionally cloning if missing.

        Convention (from repo_checkout.py):
            repos/{owner}__{name}/  e.g. repos/jqlang__jq/
        """
        dir_name = instance.repo.replace("/", "__")
        repo_root = self.repos_dir / dir_name

        if not repo_root.exists():
            if self.auto_checkout:
                logger.info("[%s] Repo not found at %s — cloning at base_commit=%s",
                            instance.instance_id, repo_root, instance.base_commit[:10])
                repo_root = ensure_repo_at_commit(
                    instance.repo,
                    instance.base_commit,
                    self.repos_dir,
                )
            else:
                raise FileNotFoundError(
                    f"Repo not found at {repo_root}. "
                    f"Run ensure_repo_at_commit('{instance.repo}', '{instance.base_commit}') "
                    f"or set auto_checkout=True."
                )
        return repo_root

    # ── Keyword collection ────────────────────────────────────────────────────

    @staticmethod
    def _collect_keywords(plan: PlannerOutput) -> list[str]:
        """Merge and deduplicate all keyword signals from the Planner.

        Combines ``plan.keywords`` and ``plan.priority_functions`` while
        preserving order (priority_functions first for grep, keywords for graph).
        """
        seen: set[str] = set()
        result: list[str] = []
        # Priority functions go first — they are the highest-signal terms
        for kw in plan.priority_functions + plan.keywords:
            kw_clean = kw.strip()
            if kw_clean and kw_clean not in seen:
                seen.add(kw_clean)
                result.append(kw_clean)
        return result

    # ── Optional LLM re-ranking ───────────────────────────────────────────────

    def _llm_rerank(
        self,
        candidates: list[LocalizerCandidate],
        instance: SWEInstance,
    ) -> list[LocalizerCandidate]:
        """Re-rank candidates using a Voyager / Qwen3-30B call.

        Enabled when ``use_llm_rerank=True`` (for V6+ ablation).
        Falls back to original order on any error.
        """
        try:
            import json as _json
            from agents.llm import DEFAULT_MODEL, chat, make_client

            client = make_client(api_key=self._api_key, base_url=self._base_url)
            file_list = "\n".join(
                f"{i+1}. {c.file_path}  (score={c.score:.3f}, {c.reason})"
                for i, c in enumerate(candidates)
            )
            user = (
                f"Repository: {instance.instance_id}\n\n"
                f"Bug report:\n{instance.problem_statement[:800]}\n\n"
                f"Candidate files for the bug:\n{file_list}\n\n"
                "Re-rank these files from most to least likely to contain the bug. "
                "Return ONLY a JSON array of file paths in ranked order. "
                "Example: [\"src/jv.c\", \"src/execute.c\"]"
            )
            text = chat(
                client=client,
                model=self._model_name or DEFAULT_MODEL,
                system="You rank candidate source files for a bug localisation task. Output JSON only.",
                user=user,
                temperature=0.0,
                max_tokens=512,
                max_retries=1,
                agent_name="Localizer.rerank",
            ).strip()
            if text.startswith("```"):
                text = "\n".join(
                    ln for ln in text.splitlines() if not ln.startswith("```")
                ).strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("no JSON array in re-rank response")
            ranked_paths: list[str] = _json.loads(text[start:end])

            path_to_cand = {c.file_path: c for c in candidates}
            reranked = [path_to_cand[p] for p in ranked_paths if p in path_to_cand]
            seen_paths = set(ranked_paths)
            for c in candidates:
                if c.file_path not in seen_paths:
                    reranked.append(c)
            return reranked

        except Exception as exc:
            logger.warning("LLM re-ranking failed (%s) — keeping original order", exc)
            return candidates
