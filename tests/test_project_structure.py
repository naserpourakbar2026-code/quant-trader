from pathlib import Path

from src.core.config import PROJECT_ROOT

EXPECTED_DIRS = [
    "config",
    "data/raw",
    "data/processed",
    "data/cache",
    "src/data",
    "src/features",
    "src/strategies",
    "src/risk",
    "src/backtest",
    "src/optimization",
    "src/walkforward",
    "src/montecarlo",
    "src/portfolio",
    "src/execution",
    "src/brokers",
    "src/reporting",
    "src/core",
    "tests",
    "notebooks",
    "reports",
    "logs",
    "scripts",
]

EXPECTED_FILES = [
    "main.py",
    "requirements.txt",
    ".env.example",
    "README.md",
    "CLAUDE.md",
    "pyproject.toml",
    "config/settings.yaml",
    "config/strategies.yaml",
    "config/brokers.yaml",
]


def test_expected_directories_exist():
    missing = [d for d in EXPECTED_DIRS if not (PROJECT_ROOT / d).is_dir()]
    assert not missing, f"Missing expected directories: {missing}"


def test_expected_files_exist():
    missing = [f for f in EXPECTED_FILES if not (PROJECT_ROOT / f).is_file()]
    assert not missing, f"Missing expected files: {missing}"


def test_env_is_gitignored():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text()
    assert ".env" in gitignore
