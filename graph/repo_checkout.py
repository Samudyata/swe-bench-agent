"""Git repository clone and checkout helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

# GitHub URLs for the 3 SWE-bench-C repositories
REPO_URLS = {
    "facebook/zstd": "https://github.com/facebook/zstd.git",
    "jqlang/jq": "https://github.com/jqlang/jq.git",
    "redis/redis": "https://github.com/redis/redis.git",
}

DEFAULT_REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(
        ["git"] + args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_repo_at_commit(
    repo_slug: str,
    base_commit: str,
    repos_dir: Path | str | None = None,
) -> Path:
    """Clone repo if needed and checkout a specific commit.

    Args:
        repo_slug: e.g. "jqlang/jq"
        base_commit: git commit hash to check out
        repos_dir: parent directory for cloned repos

    Returns:
        Path to the repo root at the requested commit.
    """
    repos_dir = Path(repos_dir) if repos_dir else DEFAULT_REPOS_DIR
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Repo stored as owner__name
    dir_name = repo_slug.replace("/", "__")
    repo_path = repos_dir / dir_name

    if not repo_path.exists():
        url = REPO_URLS.get(repo_slug)
        if not url:
            url = f"https://github.com/{repo_slug}.git"
        print(f"Cloning {repo_slug} ...")
        _run_git(["clone", url, str(repo_path)])

    # Checkout the requested commit
    _run_git(["checkout", base_commit, "--force"], cwd=repo_path)
    _run_git(["clean", "-fdx"], cwd=repo_path)

    return repo_path
