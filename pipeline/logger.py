"""Structured JSONL logger for inter-agent pipeline events.

Every message exchanged between agents is recorded here as a single JSON
line in a per-run log file. This gives full reproducibility and enables
offline analysis of agent behaviour.

Log file location:
    logs/{instance_id}_{YYYYMMDD_HHMMSS}.jsonl

Each line is a JSON object:
    {
        "timestamp":   "2024-01-01T12:00:00.000Z",
        "instance_id": "jq-493__jqlang__jq",
        "event":       "planner_output",
        "agent":       "planner",
        "data":        { ... }
    }

Standard event names
--------------------
instance_start       controller   Pipeline begins for an instance
planner_output       planner      PlannerOutput produced
localizer_output     localizer    ContextBundle produced
diagnostician_output diagnostician FixPlan produced
patch_output         patcher      PatchOutput produced
validation_result    validator    ValidationResult produced
retry_routed         controller   FeedbackMessage sent to an agent
instance_end         controller   Final result (resolved / failed)
error                *            Unexpected exception caught
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


class PipelineLogger:
    """JSONL logger for one pipeline run (one SWE-bench instance)."""

    def __init__(
        self,
        instance_id: str,
        log_dir: Path | str = Path("logs"),
        *,
        echo: bool = True,
    ) -> None:
        """
        Args:
            instance_id: The SWE-bench instance being processed.
            log_dir:     Directory where log files are written.
            echo:        If True, also print a compact summary line to stdout.
        """
        self.instance_id = instance_id
        self.echo = echo
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"{instance_id}_{timestamp}.jsonl"

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, event: str, agent: str, data: dict) -> None:
        """Write one structured log entry.

        Args:
            event:  Short identifier for what happened (e.g. "planner_output").
            agent:  Name of the agent that produced the event.
            data:   Serialisable dict with the event payload.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "instance_id": self.instance_id,
            "event": event,
            "agent": agent,
            "data": data,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.echo:
            self._print_summary(event, agent, data)

    def log_error(self, agent: str, exc: Exception, context: dict | None = None) -> None:
        """Convenience method to log an unexpected exception."""
        self.log(
            event="error",
            agent=agent,
            data={
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "context": context or {},
            },
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _print_summary(self, event: str, agent: str, data: dict) -> None:
        """Print a compact one-liner to stdout for live monitoring."""
        prefix = f"[{self.instance_id}] [{agent}] {event}"
        # Add a short human-readable tail depending on event type
        tail = ""
        if event == "planner_output":
            kws = data.get("keywords", [])[:4]
            tail = f"  keywords={kws}"
        elif event == "localizer_output":
            conf = data.get("confidence", "?")
            n = len(data.get("candidates", []))
            tail = f"  confidence={conf:.2f}  candidates={n}"
        elif event == "patch_output":
            lines = data.get("unified_diff", "").count("\n")
            tail = f"  diff_lines={lines}"
        elif event == "validation_result":
            status = data.get("status", "?")
            resolved = data.get("resolved", False)
            tail = f"  status={status}  resolved={resolved}"
        elif event == "retry_routed":
            route_to = data.get("route_to", "?")
            num = data.get("retry_number", "?")
            tail = f"  route_to={route_to}  attempt={num}"
        elif event == "error":
            tail = f"  {data.get('exception_type')}: {data.get('message', '')[:80]}"
        print(prefix + tail, flush=True)

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def read_all(self) -> list[dict]:
        """Read all entries from the log file (useful for post-hoc analysis)."""
        entries = []
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return entries

    @staticmethod
    def load(path: Path | str) -> list[dict]:
        """Load a JSONL log file from disk (utility for offline analysis)."""
        entries = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries
