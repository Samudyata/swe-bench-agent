"""Environment-driven configuration for the swe-bench-agent pipeline.

All settings have sensible defaults.  Override via environment variables
or by passing keyword arguments to Config().

Environment variables
---------------------
GEMINI_API_KEY      Required. Google Gemini API key.
MODEL_NAME          Gemini model (default: gemini-2.0-flash).
CONF_THRESHOLD      Localizer confidence threshold (default: 0.4).
MAX_RETRIES         Max retries per agent on failure (default: 2).
LOG_DIR             Directory for JSONL logs (default: logs/).
GRAPHS_DIR          Directory with pre-built graph JSON files (default: graphs/).
REPOS_DIR           Directory with cloned repositories (default: repos/).
RESULTS_DIR         Directory for result JSON outputs (default: results/).

Example .env file
-----------------
    GEMINI_API_KEY=your-key-here
    MODEL_NAME=gemini-2.0-flash
    CONF_THRESHOLD=0.4
    MAX_RETRIES=2
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Pipeline configuration with environment-variable overrides."""

    # LLM settings
    model_name: str = "gemini-2.0-flash"
    gemini_api_key: str = ""

    # Pipeline behaviour
    confidence_threshold: float = 0.4   # below this → route LOW_CONF to Planner
    max_retries: int = 2                 # max retries per agent slot

    # Filesystem paths
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    graphs_dir: Path = field(default_factory=lambda: Path("graphs"))
    repos_dir: Path = field(default_factory=lambda: Path("repos"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self) -> None:
        # Ensure Path objects
        self.log_dir = Path(self.log_dir)
        self.graphs_dir = Path(self.graphs_dir)
        self.repos_dir = Path(self.repos_dir)
        self.results_dir = Path(self.results_dir)

    @classmethod
    def from_env(cls) -> Config:
        """Build Config from environment variables (+ optional .env file)."""
        _load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            import warnings
            warnings.warn(
                "GEMINI_API_KEY is not set. Planner LLM calls will fail.",
                stacklevel=2,
            )
        return cls(
            gemini_api_key=api_key,
            model_name=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
            confidence_threshold=float(os.getenv("CONF_THRESHOLD", "0.4")),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
            log_dir=Path(os.getenv("LOG_DIR", "logs")),
            graphs_dir=Path(os.getenv("GRAPHS_DIR", "graphs")),
            repos_dir=Path(os.getenv("REPOS_DIR", "repos")),
            results_dir=Path(os.getenv("RESULTS_DIR", "results")),
        )

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in (self.log_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    """Load .env file if python-dotenv is available (optional dependency)."""
    try:
        from dotenv import load_dotenv
        # Walk up from CWD to find a .env file
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents):
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
    except ImportError:
        pass   # dotenv not installed — rely purely on os.environ
