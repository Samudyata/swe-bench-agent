#!/usr/bin/env python3
"""
run_experiments.py — Run ablation experiments V4-V7 on SWE-bench-C predictions.

Reads a predictions JSONL file (instance_id + model_patch) and validates
each patch through the ValidatorAgent. Tracks success at each stage:
  - Localization (patch applies)
  - Compilation (make succeeds)
  - Resolution (all tests pass)

Usage:
    python scripts/run_experiments.py \
        --predictions path/to/predictions.jsonl \
        --run-name V4 \
        --repos-dir path/to/repos \
        --dataset path/to/swebench_c.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.validator import validate
from agents.schemas import PatchResult, FixPlan
from graph.repo_checkout import ensure_repo_at_commit


# ── Data loading ──────────────────────────────────────────────────────────────

def load_predictions(predictions_path: Path) -> list[dict]:
    """
    Load predictions JSONL file.

    Each line should contain:
      - instance_id: str
      - model_patch: str (unified diff)

    Returns:
        List of prediction dicts
    """
    predictions = []
    with open(predictions_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pred = json.loads(line)
                predictions.append(pred)
            except json.JSONDecodeError as e:
                print(f"  [run_experiments] WARNING: Line {line_num} invalid JSON: {e}")
    return predictions


def load_dataset(dataset_path: Path) -> dict[str, dict]:
    """
    Load SWE-bench-C dataset JSON.

    Returns:
        Dict mapping instance_id -> instance dict
    """
    with open(dataset_path, encoding="utf-8") as f:
        instances_list = json.load(f)

    # Convert to lookup dict
    instances = {}
    for inst in instances_list:
        instance_id = inst["instance_id"]
        instances[instance_id] = inst

    return instances


# ── Result tracking ───────────────────────────────────────────────────────────

class ExperimentStats:
    """Track cumulative statistics across all instances."""

    def __init__(self):
        self.total = 0
        self.apply_ok = 0
        self.compile_ok = 0
        self.tests_ok = 0
        self.resolved = 0

    def update(self, result: dict) -> None:
        """Update stats based on a validation result."""
        self.total += 1

        if result.get("apply_ok", False):
            self.apply_ok += 1

        if result.get("compile_ok", False):
            self.compile_ok += 1

        if result.get("tests_ok", False):
            self.tests_ok += 1

        if result.get("resolved", False):
            self.resolved += 1

    def print_summary(self) -> None:
        """Print current statistics."""
        if self.total == 0:
            print("  [Stats] No instances processed yet")
            return

        loc_pct = 100.0 * self.apply_ok / self.total
        comp_pct = 100.0 * self.compile_ok / self.total
        res_pct = 100.0 * self.resolved / self.total

        print(f"  [Stats] {self.total} instances | "
              f"Localization: {self.apply_ok}/{self.total} ({loc_pct:.1f}%) | "
              f"Compilation: {self.compile_ok}/{self.total} ({comp_pct:.1f}%) | "
              f"Resolution: {self.resolved}/{self.total} ({res_pct:.1f}%)")


# ── Patch validation ──────────────────────────────────────────────────────────

def create_patch_result(instance_id: str, model_patch: str) -> PatchResult:
    """
    Create a PatchResult from a prediction.

    Since we're running standalone validation (not the full agent pipeline),
    we create a minimal FixPlan and PatchResult.

    Args:
        instance_id: SWE-bench instance ID
        model_patch: Unified diff from the model

    Returns:
        PatchResult ready for validation
    """
    # Create a minimal FixPlan
    fix_plan = FixPlan(
        issue_id=instance_id,
        root_cause="(from predictions file)",
        fix_description="(from predictions file)",
        affected_files=[],
        affected_regions=[],
        test_constraints=[],
        suggested_approach="(from predictions file)",
        confidence=1.0,
    )

    # Create PatchResult
    patch_result = PatchResult(
        issue_id=instance_id,
        patch=model_patch,
        modified_files=[],
        fix_plan=fix_plan,
        patcher_notes="standalone_validation",
        attempt_number=1,
    )

    return patch_result


def validate_instance(
    prediction: dict,
    instance: dict,
    repos_dir: Path,
) -> Optional[dict]:
    """
    Validate a single prediction.

    Args:
        prediction: Dict with instance_id and model_patch
        instance: SWE-bench instance dict
        repos_dir: Directory containing repositories

    Returns:
        Validation result dict, or None if validation failed to run
    """
    instance_id = prediction["instance_id"]
    model_patch = prediction.get("model_patch", "")

    if not model_patch:
        print(f"  [run_experiments] WARNING: No model_patch for {instance_id}")
        return None

    # Checkout repository at base commit
    repo = instance["repo"]
    base_commit = instance["base_commit"]

    try:
        repo_root = ensure_repo_at_commit(repo, base_commit, repos_dir)
    except Exception as e:
        print(f"  [run_experiments] ERROR: Could not checkout {repo} at {base_commit}: {e}")
        return None

    # Create PatchResult
    patch_result = create_patch_result(instance_id, model_patch)

    # Run validation
    try:
        result = validate(patch_result, str(repo_root), instance)
        return result
    except Exception as e:
        print(f"  [run_experiments] ERROR: Validation failed for {instance_id}: {e}")
        return {
            "issue_id": instance_id,
            "status": "validation_error",
            "error_output": str(e),
            "resolved": False,
            "apply_ok": False,
        }


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiments(
    predictions_path: Path,
    dataset_path: Path,
    repos_dir: Path,
    run_name: str,
    results_dir: Optional[Path] = None,
) -> None:
    """
    Run validation experiments on all predictions.

    Args:
        predictions_path: Path to predictions JSONL file
        dataset_path: Path to SWE-bench-C dataset JSON
        repos_dir: Directory containing repositories
        run_name: Name for this experimental run (e.g., "V4")
        results_dir: Directory to save results (default: PROJECT_ROOT/results)
    """
    # Setup results directory
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results"
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_experiments] Starting experiment: {run_name}")
    print(f"[run_experiments] Results will be saved to: {run_dir}")

    # Load data
    print(f"[run_experiments] Loading predictions from {predictions_path}")
    predictions = load_predictions(predictions_path)
    print(f"[run_experiments] Loaded {len(predictions)} predictions")

    print(f"[run_experiments] Loading dataset from {dataset_path}")
    instances = load_dataset(dataset_path)
    print(f"[run_experiments] Loaded {len(instances)} instances")

    # Initialize stats
    stats = ExperimentStats()

    # Process each prediction
    for i, prediction in enumerate(predictions, start=1):
        instance_id = prediction["instance_id"]
        print(f"\n[run_experiments] [{i}/{len(predictions)}] Processing {instance_id}")

        # Get instance from dataset
        if instance_id not in instances:
            print(f"  [run_experiments] WARNING: {instance_id} not in dataset, skipping")
            continue

        instance = instances[instance_id]

        # Run validation
        t0 = time.time()
        result = validate_instance(prediction, instance, repos_dir)
        elapsed = time.time() - t0

        if result is None:
            print(f"  [run_experiments] Skipped {instance_id}")
            continue

        # Update stats
        stats.update(result)

        # Save result
        result_file = run_dir / f"{instance_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            # Convert PatchResult to dict if present
            result_to_save = result.copy()
            if "patch_result" in result_to_save:
                result_to_save["patch_result"] = result_to_save["patch_result"].to_dict()
            json.dump(result_to_save, f, indent=2)

        # Print result
        status = result.get("status", "unknown")
        print(f"  [run_experiments] Result: {status} ({elapsed:.1f}s)")

        # Print running stats every 10 instances
        if i % 10 == 0:
            stats.print_summary()

    # Final summary
    print("\n" + "=" * 80)
    print(f"[run_experiments] Experiment {run_name} complete!")
    print("=" * 80)
    stats.print_summary()
    print(f"\nResults saved to: {run_dir}")

    # Save summary
    summary = {
        "run_name": run_name,
        "total": stats.total,
        "apply_ok": stats.apply_ok,
        "compile_ok": stats.compile_ok,
        "tests_ok": stats.tests_ok,
        "resolved": stats.resolved,
        "localization_pct": 100.0 * stats.apply_ok / stats.total if stats.total > 0 else 0.0,
        "compilation_pct": 100.0 * stats.compile_ok / stats.total if stats.total > 0 else 0.0,
        "resolution_pct": 100.0 * stats.resolved / stats.total if stats.total > 0 else 0.0,
    }

    summary_file = run_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run validation experiments on SWE-bench-C predictions"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name for this experimental run (e.g., V4, V5)",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=PROJECT_ROOT / "repos",
        help="Directory containing repositories (default: PROJECT_ROOT/repos)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "swe-bench-c.json",
        help="Path to SWE-bench-C dataset JSON (default: PROJECT_ROOT/data/swe-bench-c.json)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: PROJECT_ROOT/results)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.predictions.exists():
        print(f"ERROR: Predictions file not found: {args.predictions}")
        sys.exit(1)

    if not args.dataset.exists():
        print(f"ERROR: Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Run experiments
    run_experiments(
        predictions_path=args.predictions,
        dataset_path=args.dataset,
        repos_dir=args.repos_dir,
        run_name=args.run_name,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
