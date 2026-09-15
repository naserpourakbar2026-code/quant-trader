"""Reproducibility metadata (CLAUDE.md Section 36).

Every experiment/backtest/optimization/walk-forward/Monte-Carlo result
should record enough to reproduce it: git commit hash, data version,
configuration, parameters, random seed, Python version, library versions.
This module provides the environment-level pieces (git/Python/library
versions); data version comes from src.data.ingestion, config/parameters/
seed from the caller.
"""
from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from src.core.config import PROJECT_ROOT

TRACKED_LIBRARIES = ["pandas", "numpy", "vectorbt", "scipy", "pydantic"]


def git_commit_hash() -> str:
    """The current commit hash, or "unknown" outside a git repo / if git
    isn't available — never fabricated."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def python_version() -> str:
    return platform.python_version()


def library_versions(libraries: list[str] | None = None) -> dict[str, str]:
    versions: dict[str, str] = {}
    for lib in libraries or TRACKED_LIBRARIES:
        try:
            versions[lib] = _pkg_version(lib)
        except PackageNotFoundError:
            versions[lib] = "unknown"
    return versions
