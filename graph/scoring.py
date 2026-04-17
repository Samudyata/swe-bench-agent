"""Keyword-weighted scoring and evidence subgraph extraction."""

from __future__ import annotations

import re
from collections import defaultdict

from graph.model import DepGraph
from graph.query import expand_hop

# Common C / English stopwords to filter out
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "this", "that", "these", "those", "it", "its", "if", "then", "else",
    "when", "where", "which", "what", "who", "whom", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "only", "own", "same", "than", "too", "very", "just", "because",
    # Common C terms that are too generic
    "int", "char", "void", "return", "include", "define", "struct",
    "const", "static", "unsigned", "signed", "long", "short", "double",
    "float", "sizeof", "typedef", "enum", "union", "extern", "inline",
    "null", "true", "false", "error", "file", "function", "variable",
}

_IDENT_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def extract_keywords(text: str) -> list[str]:
    """Extract C-identifier-like keywords from natural language text."""
    tokens = _IDENT_RE.findall(text.lower())
    seen = set()
    result = []
    for t in tokens:
        if t not in _STOPWORDS and len(t) > 2 and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def keyword_search(
    graph: DepGraph, keywords: list[str]
) -> list[tuple[str, float]]:
    """Score graph nodes by keyword relevance.

    Matches keywords against node names and file paths.
    Returns (node_id, score) sorted descending by score.
    """
    scores: dict[str, float] = defaultdict(float)

    for nid, node in graph.nodes.items():
        text_parts = [node["path"].lower()]
        if node["kind"] == "function":
            text_parts.append(node["name"].lower())
            # Split camelCase / snake_case for matching
            text_parts.extend(_split_identifier(node["name"]))

        node_text = " ".join(text_parts)
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in node_text:
                # Exact function name match is worth more
                if node["kind"] == "function" and kw_lower == node["name"].lower():
                    scores[nid] += 3.0
                elif node["kind"] == "function" and kw_lower in node["name"].lower():
                    scores[nid] += 2.0
                else:
                    scores[nid] += 1.0

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked


def _split_identifier(name: str) -> list[str]:
    """Split a C identifier into parts (snake_case and camelCase)."""
    # snake_case
    parts = name.lower().split("_")
    # camelCase
    parts.extend(p.lower() for p in re.findall(r"[A-Z][a-z]+", name))
    return [p for p in parts if len(p) > 2]


def evidence_subgraph(
    graph: DepGraph,
    keywords: list[str],
    top_k: int = 5,
    hops: int = 2,
    min_confidence: float = 0.3,
) -> set[str]:
    """Build an evidence subgraph: find keyword-relevant seeds and expand.

    This is the "smarter than static" traversal: instead of returning all
    neighbors blindly, it starts from keyword-scored seed nodes and expands
    only through edges above a confidence threshold.

    Returns: set of node IDs in the evidence subgraph.
    """
    ranked = keyword_search(graph, keywords)
    if not ranked:
        return set()

    seeds = {nid for nid, _ in ranked[:top_k]}
    return expand_hop(graph, seeds, hops=hops, min_confidence=min_confidence)
