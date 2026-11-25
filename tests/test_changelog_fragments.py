"""Changelog fragments in changelog.d follow the naming in changelog.d/README.md (#180)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAGMENTS = ROOT / "changelog.d"
TYPES = ("added", "changed", "deprecated", "removed", "fixed", "security")
NAME = re.compile(rf"^(\d+|\+[a-z0-9][a-z0-9-]*)\.({'|'.join(TYPES)})(\.\d+)?\.md$")


def fragments():
    return [p for p in FRAGMENTS.iterdir() if p.name != "README.md"]


def test_fragment_names():
    bad = [p.name for p in fragments() if not NAME.match(p.name)]
    assert not bad, f"rename to <issue>.<type>.md or +<slug>.<type>.md: {bad}"


def test_fragments_are_not_empty():
    empty = [p.name for p in fragments() if not p.read_text().strip()]
    assert not empty


def test_fragment_types_match_towncrier_config():
    config = (ROOT / "pyproject.toml").read_text()
    configured = re.findall(
        r'\[\[tool\.towncrier\.type\]\]\s*directory = "(\w+)"', config
    )
    assert tuple(configured) == TYPES


def test_changelog_has_the_towncrier_marker():
    assert (
        "<!-- towncrier release notes start -->" in (ROOT / "CHANGELOG.md").read_text()
    )
