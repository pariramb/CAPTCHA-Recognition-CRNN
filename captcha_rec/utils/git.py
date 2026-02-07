from __future__ import annotations

import subprocess


def get_git_commit_id(default: str = "unknown") -> str:
    """
    Returns current git commit hash or 'unknown' if not available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit if commit else default
    except Exception:
        return default
