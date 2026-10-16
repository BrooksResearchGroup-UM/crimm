"""Shared fixtures for the crimm test suite.

The default run is offline: every structure comes from ``tests/data``. Tests that need
the network, pyCHARMM or the cgenff executable carry a marker and are deselected unless
asked for (see ``[tool.pytest.ini_options]`` in ``pyproject.toml``).
"""
import os
import shutil
import warnings
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"

# Values read off each entry. Heavy-atom counts are what the parser returns with
# hydrogens excluded and the first altloc selected.
EXPECTED = {
    "1ubq": {
        "chains": [("A", "Polypeptide(L)", 76), ("B", "Solvent", 58)],
        "n_atoms": 660,
        "polymer_charge": {"A": 0},
    },
    "1crn": {
        "chains": [("A", "Polypeptide(L)", 46)],
        "n_atoms": 327,
        "polymer_charge": {"A": 0},
    },
    "2igd": {
        "chains": [("A", "Polypeptide(L)", 61), ("B", "Solvent", 106)],
        "n_atoms": 574,
        "polymer_charge": {"A": -2},
    },
    "3ptb": {
        "chains": [
            ("A", "Polypeptide(L)", 223),
            ("B", "Heterogens", 1),
            ("C", "Heterogens", 1),
            ("D", "Solvent", 62),
        ],
        "n_atoms": 1701,
        "polymer_charge": {"A": 6},
    },
    "1bna": {
        "chains": [
            ("A", "Polydeoxyribonucleotide", 12),
            ("B", "Polydeoxyribonucleotide", 12),
            ("C", "Solvent", 37),
            ("D", "Solvent", 43),
        ],
        "n_atoms": 566,
        "polymer_charge": {"A": -12, "B": -12},
    },
}
ALL_IDS = sorted(EXPECTED)


def parse_structure(pdb_id, **parser_kwargs):
    """Parse a fixture mmCIF file with the same options ``fetch_rcsb`` uses."""
    from crimm.IO import MMCIFParser

    options = dict(
        first_model_only=True,
        use_bio_assembly=True,
        include_solvent=True,
        include_hydrogens=False,
        QUIET=True,
    )
    options.update(parser_kwargs)
    return MMCIFParser(**options).get_structure(str(DATA_DIR / f"{pdb_id}.cif"))


def make_organized_model(pdb_id):
    """Fresh ``OrganizedModel`` built offline from a fixture."""
    from crimm.StructEntities.OrganizedModel import OrganizedModel

    return OrganizedModel(parse_structure(pdb_id).models[0])


def make_prepared_model(pdb_id):
    """Fresh ``OrganizedModel`` with CHARMM topology generated (no CGenFF)."""
    from crimm.Modeller import TopologyGenerator

    model = make_organized_model(pdb_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TopologyGenerator().generate_model(model, QUIET=True)
    return model


@pytest.fixture
def load_structure():
    """Factory: ``load_structure('1ubq', include_solvent=False)`` -> Structure."""
    return parse_structure


@pytest.fixture
def organized_model():
    """Factory returning a fresh ``OrganizedModel``; safe to modify."""
    return make_organized_model


@pytest.fixture
def prepared_model():
    """Factory returning a fresh model with topology; safe to modify."""
    return make_prepared_model


@pytest.fixture(scope="session")
def shared_prepared():
    """Session cache of prepared models. Treat the result as read-only."""
    cache = {}

    def get(pdb_id):
        if pdb_id not in cache:
            cache[pdb_id] = make_prepared_model(pdb_id)
        return cache[pdb_id]

    return get


@pytest.fixture(scope="session")
def cgenff_path():
    """Path to the cgenff executable; skips the test when it cannot be found."""
    path = os.environ.get("CRIMM_CGENFF_PATH") or shutil.which("cgenff")
    if not path or not os.path.exists(path):
        pytest.skip("cgenff executable not found (set CRIMM_CGENFF_PATH)")
    return path
