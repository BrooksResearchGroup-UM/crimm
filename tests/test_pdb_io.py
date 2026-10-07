"""PDB string output and the PDB parser."""
import numpy as np
import pytest

from crimm.IO import PDBParser
from crimm.IO.PDBString import get_pdb_str


def reparse(text, tmp_path):
    path = tmp_path / "out.pdb"
    path.write_text(text)
    return PDBParser(QUIET=True).get_structure(str(path))


def test_atom_line_format(load_structure):
    residue = load_structure("1ubq").models[0]["A"][1]
    lines = get_pdb_str(residue).splitlines()
    assert lines[0] == (
        "ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67"
        "           N  "
    )
    assert all(len(line) == 80 for line in lines if line.startswith("ATOM"))
    assert lines[-2].startswith("TER")
    assert lines[-1] == "END"


def test_one_line_per_atom(load_structure):
    model = load_structure("1ubq").models[0]
    lines = get_pdb_str(model).splitlines()
    records = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    assert len(records) == 660
    assert sum(line.startswith("TER") for line in lines) == 2


def test_round_trip_of_parsed_structure(load_structure, tmp_path):
    model = load_structure("1ubq").models[0]
    original = np.array([atom.coord for atom in model.get_atoms()])
    reread = reparse(get_pdb_str(model), tmp_path)
    chains = [(c.id, c.chain_type, len(c)) for c in reread.models[0]]
    assert chains == [("A", "Polypeptide(L)", 76), ("B", "Solvent", 58)]
    coords = np.array([atom.coord for atom in reread.get_atoms()])
    assert coords == pytest.approx(original, abs=1e-3)


def test_round_trip_after_topology_with_water_conversion(shared_prepared, tmp_path):
    model = shared_prepared("1ubq")
    reread = reparse(get_pdb_str(model, convert_water=True), tmp_path)
    assert len(list(reread.get_atoms())) == len(list(model.get_atoms()))
    assert [c.chain_type for c in reread.models[0]] == ["Polypeptide(L)", "Solvent"]


@pytest.mark.xfail(
    strict=True,
    reason="Default get_pdb_str writes 4-character TIP3 residue names, which shifts "
    "the chain column; crimm's own PDBParser then fails on the output",
)
def test_round_trip_after_topology_default_options(shared_prepared, tmp_path):
    model = shared_prepared("1ubq")
    reread = reparse(get_pdb_str(model), tmp_path)
    assert len(list(reread.get_atoms())) == len(list(model.get_atoms()))


def test_include_alt_writes_all_locations(load_structure):
    model = load_structure("2igd").models[0]

    def n_records(text):
        return sum(line.startswith(("ATOM", "HETATM")) for line in text.splitlines())

    assert n_records(get_pdb_str(model)) == 574
    assert n_records(get_pdb_str(model, include_alt=True)) == 606
