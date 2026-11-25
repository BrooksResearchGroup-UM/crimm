"""crimm's version strings agree (#55)."""

import re
from importlib.metadata import version
from pathlib import Path

import crimm

ROOT = Path(__file__).resolve().parents[1]


def _assignment(path, name):
    match = re.search(
        rf"^{name}\s*=\s*[\"']([^\"']+)[\"']", path.read_text(), re.MULTILINE
    )
    assert match, f"no {name} in {path}"
    return match.group(1)


def test_version_matches_installed_metadata():
    assert crimm.__version__ == version("crimm")


def test_docs_release_matches_pyproject():
    pyproject = _assignment(ROOT / "pyproject.toml", "version")
    assert _assignment(ROOT / "docs" / "conf.py", "release") == pyproject
