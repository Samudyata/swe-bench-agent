#!/usr/bin/env python3
"""CLI entry point for the swe-bench-agent pipeline.

Examples
--------
Run one specific instance:
    python run_pipeline.py --instance_id jq-493__jqlang__jq

Run all instances for one repo:
    python run_pipeline.py --repo jqlang/jq

Run all 179 SWE-bench-C instances:
    python run_pipeline.py --all

Dry-run (Planner only, no downstream agents needed):
    python run_pipeline.py --instance_id jq-493__jqlang__jq --planner-only

Use stub agents for end-to-end smoke testing:
    python run_pipeline.py --instance_id jq-493__jqlang__jq --stub-agents
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from graph.model import DepGraph
from pipeline.schema import SWEInstance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_instances(config: Config) -> list[dict]:
    """Load all SWE-bench-C instances from a local JSON file or HuggingFace."""
    candidates = [
        PROJECT_ROOT.parent / "SWE-bench-c" / "gold.validate-gold.json",
        PROJECT_ROOT / "data" / "swe-bench-c.json",
    ]
    for path in candidates:
        if path.exists():
            log.info("Loading instances from %s", path)
            with open(path, encoding="utf-8") as f:
                return json.load(f)

    try:
        from datasets import load_dataset
        log.info("Loading SWE-bench-C from HuggingFace…")
        ds = load_dataset("xingyaoww/swe-bench-c", split="test")
        return list(ds)
    except ImportError:
        pass

    log.error(
        "Cannot find SWE-bench-C dataset.\n"
        "  Option 1: place data at %s\n"
        "  Option 2: pip install datasets",
        candidates[1],
    )
    sys.exit(1)


def filter_instances(
    all_instances: list[dict],
    instance_id: str | None,
    repo: str | None,
) -> list[dict]:
    """Apply --instance_id / --repo filter."""
    if instance_id:
        matched = [i for i in all_instances if i["instance_id"] == instance_id]
        if not matched:
            log.error("instance_id %r not found in dataset", instance_id)
            sys.exit(1)
        return matched
    if repo:
        matched = [i for i in all_instances if i.get("repo") == repo]
        if not matched:
            log.error("No instances found for repo %r", repo)
            sys.exit(1)
        log.info("Found %d instances for repo %s", len(matched), repo)
        return matched
    return all_instances


# ─────────────────────────────────────────────────────────────────────────────
# Agent construction
# ─────────────────────────────────────────────────────────────────────────────

def build_agents(config: Config, stub_mode: bool):
    """Instantiate all agent objects.

    Returns a dict with keys: planner, localizer, diagnostician, patcher, validator.
    When stub_mode=True, stub agents are used for everything except the Planner.
    All three LLM-backed agents (Planner, Diagnostician, Patcher) talk to
    Voyager (OpenAI-compatible) via agents/llm.py.
    """
    from agents.planner import PlannerAgent
    from agents.stubs import DiagnosticianStub, LocalizerStub, PatcherStub, ValidatorStub

    llm_kwargs = dict(
        api_key=config.openai_api_key or None,
        base_url=config.openai_api_base or None,
    )

    planner = PlannerAgent(model_name=config.model_name, **llm_kwargs)

    if stub_mode:
        log.warning("Running with STUB agents for Person 3/4/5 — results are not meaningful")
        return dict(
            planner=planner,
            localizer=LocalizerStub(stub_mode=True),
            diagnostician=DiagnosticianStub(stub_mode=True),
            patcher=PatcherStub(stub_mode=True),
            validator=ValidatorStub(stub_mode=True),
        )

    try:
        from agents.localizer import LocalizerAgent
        localizer = LocalizerAgent()
    except ImportError:
        log.warning("agents/localizer.py not found — using stub")
        localizer = LocalizerStub(stub_mode=False)

    try:
        from agents.diagnostician import DiagnosticianAgent
        diagnostician = DiagnosticianAgent(model_name=config.model_name, **llm_kwargs)
    except ImportError:
        log.warning("agents/diagnostician.py not found — using stub")
        diagnostician = DiagnosticianStub(stub_mode=False)

    try:
        from agents.patcher import PatcherAgent
        patcher = PatcherAgent(model_name=config.model_name, **llm_kwargs)
    except ImportError:
        log.warning("agents/patcher.py not found — using stub")
        patcher = PatcherStub(stub_mode=False)

    try:
        from agents.validator import ValidatorAgent
        validator = ValidatorAgent()
    except ImportError:
        log.warning("agents/validator.py not found — using stub")
        validator = ValidatorStub(stub_mode=False)

    return dict(
        planner=planner,
        localizer=localizer,
        diagnostician=diagnostician,
        patcher=patcher,
        validator=validator,
    )


def _resolve_repo_root(config: Config, instance: SWEInstance) -> Path:
    """Return the filesystem path where this instance's repo should live.

    Layout: <repos_dir>/<owner>__<name>/
    """
    repo_slug = instance.repo.replace("/", "__")
    return config.repos_dir / repo_slug


# ─────────────────────────────────────────────────────────────────────────────
# Main run logic
# ─────────────────────────────────────────────────────────────────────────────

def run_instance(
    raw: dict,
    config: Config,
    agents: dict,
    planner_only: bool = False,
) -> dict:
    """Run the pipeline for one instance. Returns a result summary dict."""
    from pipeline.controller import PipelineController

    instance = SWEInstance.from_dict(raw)
    instance_id = instance.instance_id

    # Load pre-built graph
    graph_path = config.graphs_dir / f"{instance_id}.json"
    if not graph_path.exists():
        log.error("[%s] Graph not found at %s — run build_all_graphs.py first", instance_id, graph_path)
        return {"instance_id": instance_id, "error": "graph_missing"}

    graph = DepGraph.load(graph_path)
    log.info("[%s] Loaded graph: %s", instance_id, graph)

    # Point Patcher + Validator at this instance's checked-out repo
    repo_root = _resolve_repo_root(config, instance)
    for slot in ("patcher", "validator"):
        setter = getattr(agents.get(slot), "set_repo_root", None)
        if callable(setter):
            setter(str(repo_root))

    # ── Planner-only mode ─────────────────────────────────────────────────────
    if planner_only:
        from pipeline.logger import PipelineLogger
        logger_ = PipelineLogger(instance_id, config.log_dir)
        t0 = time.time()
        plan = agents["planner"].plan(instance)
        elapsed = time.time() - t0
        logger_.log("planner_output", "planner", plan.to_dict())
        print(f"[{instance_id}] Planner done in {elapsed:.1f}s")
        print(f"  issue_type:          {plan.issue_type}")
        print(f"  keywords:            {plan.keywords}")
        print(f"  priority_functions:  {plan.priority_functions}")
        print(f"  suspected_modules:   {plan.suspected_modules}")
        print(f"  search_hints:        {plan.search_hints}")
        print(f"  reasoning:           {plan.reasoning}")
        return plan.to_dict()

    # ── Full pipeline ─────────────────────────────────────────────────────────
    controller = PipelineController(
        config=config,
        planner=agents["planner"],
        localizer=agents["localizer"],
        diagnostician=agents["diagnostician"],
        patcher=agents["patcher"],
        validator=agents["validator"],
    )

    t0 = time.time()
    result = controller.run(instance, graph)
    elapsed = time.time() - t0

    status_icon = "✓" if result.resolved else "✗"
    print(f"  {status_icon} [{instance_id}] {result.status.value}  ({elapsed:.1f}s)")
    return result.to_dict()


def print_summary(results: list[dict]) -> None:
    """Print aggregate stats after a batch run."""
    total = len(results)
    resolved = sum(1 for r in results if r.get("resolved", False))
    apply_ok = sum(1 for r in results if r.get("apply_ok", False))
    compile_ok = sum(1 for r in results if r.get("compile_ok", False))
    errors = sum(1 for r in results if "error" in r)

    print("\n" + "=" * 55)
    print(f"  Total instances: {total}")
    print(f"  Errors (graph missing etc.): {errors}")
    n = total - errors
    if n > 0:
        print(f"  git apply ok:  {apply_ok}/{n}  ({100*apply_ok/n:.1f}%)")
        print(f"  compile ok:    {compile_ok}/{n}  ({100*compile_ok/n:.1f}%)")
        print(f"  RESOLVED:      {resolved}/{n}  ({100*resolved/n:.1f}%)")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="swe-bench-agent pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--instance_id", "--instance-id",
        help="Run a single instance by ID (e.g. jq-493__jqlang__jq)",
    )
    scope.add_argument(
        "--repo",
        help="Run all instances for one repo (e.g. jqlang/jq)",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Run all 179 SWE-bench-C instances",
    )

    parser.add_argument(
        "--planner-only",
        action="store_true",
        help="Only run the Planner and print its output (no downstream agents needed)",
    )
    parser.add_argument(
        "--stub-agents",
        action="store_true",
        help="Use stub implementations for Persons 3/4/5 (for end-to-end smoke testing)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name (e.g. qwen3-30b-a3b-instruct-2507)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override max retries per agent slot",
    )

    args = parser.parse_args()
    if not (args.instance_id or args.repo or args.all):
        parser.print_help()
        sys.exit(1)
    return args


def main() -> None:
    args = parse_args()
    config = Config.from_env()

    # CLI overrides
    if args.model:
        config.model_name = args.model
    if args.max_retries is not None:
        config.max_retries = args.max_retries

    config.ensure_dirs()

    # Load and filter instances
    all_instances = load_instances(config)
    instances = filter_instances(
        all_instances,
        instance_id=args.instance_id,
        repo=args.repo if not args.all else None,
    )
    log.info("Running %d instance(s)…", len(instances))

    # Build agents
    agents = build_agents(config, stub_mode=args.stub_agents)

    # Run
    results = []
    for raw in instances:
        try:
            r = run_instance(raw, config, agents, planner_only=args.planner_only)
            results.append(r)
        except NotImplementedError as exc:
            iid = raw.get("instance_id", "?")
            log.error("[%s] Agent not yet implemented: %s", iid, exc)
            results.append({"instance_id": iid, "error": str(exc)})
        except Exception as exc:
            iid = raw.get("instance_id", "?")
            log.exception("[%s] Unexpected error: %s", iid, exc)
            results.append({"instance_id": iid, "error": str(exc)})

    if not args.planner_only and len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
