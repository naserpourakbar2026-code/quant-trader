from src.core import reproducibility


def test_git_commit_hash_returns_a_real_hash_in_this_repo():
    commit_hash = reproducibility.git_commit_hash()
    assert commit_hash != "unknown"
    assert len(commit_hash) == 40
    assert all(c in "0123456789abcdef" for c in commit_hash)


def test_python_version_matches_running_interpreter():
    import sys

    version = reproducibility.python_version()
    assert version == f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def test_library_versions_returns_known_packages():
    versions = reproducibility.library_versions(["pandas", "numpy"])
    assert set(versions) == {"pandas", "numpy"}
    assert all(v != "unknown" for v in versions.values())


def test_library_versions_reports_unknown_for_missing_package():
    versions = reproducibility.library_versions(["not-a-real-package-xyz"])
    assert versions["not-a-real-package-xyz"] == "unknown"
