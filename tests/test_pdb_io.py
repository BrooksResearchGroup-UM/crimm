"""PDB string output and the PDB parser."""

import io

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


def test_round_trip_after_topology_default_options(shared_prepared, tmp_path):
    model = shared_prepared("1ubq")
    reread = reparse(get_pdb_str(model), tmp_path)
    assert len(list(reread.get_atoms())) == len(list(model.get_atoms()))
    water = reread.models[0]["B"]
    assert [c.chain_type for c in reread.models[0]] == ["Polypeptide(L)", "Solvent"]
    assert {res.resname for res in water} == {"TIP3"}
    assert [res.id for res in water] == [res.id for res in model["B"]]


def test_four_character_resname_keeps_chain_column(shared_prepared):
    lines = get_pdb_str(shared_prepared("1ubq")).splitlines()
    water = next(line for line in lines if "TIP3" in line)
    # residue name in columns 18-21, chain ID in column 22, resseq in 23-26
    assert water[17:21] == "TIP3"
    assert water[21] == "B"
    assert water[22:26] == "   1"
    assert len(water) == 80
    ter = [line for line in lines if line.startswith("TER")][-1]
    assert ter[17:21] == "TIP3"
    assert ter[21] == "B"


def test_parser_reads_four_character_resnames_in_every_model(tmp_path):
    # columns: 1-6 record, 7-11 serial, 13-16 atom, 18-21 residue, 22 chain, 23-26 resseq
    record = "HETATM    1  P   POPCL   1       0.000   0.000   0.000  1.00  0.00           P  "
    text = f"MODEL        1\n{record}\nENDMDL\nMODEL        2\n{record}\nENDMDL\nEND\n"
    path = tmp_path / "lipid.pdb"
    path.write_text(text)
    structure = PDBParser(QUIET=True, first_model_only=False).get_structure(str(path))
    residues = [res for model in structure for res in model.get_residues()]
    assert [(res.resname, res.id) for res in residues] == [
        ("POPC", ("H_POPC", 1, " ")),
        ("POPC", ("H_POPC", 1, " ")),
    ]


def _hetatm(record, serial, atom, resname, chain, resseq, element):
    return (
        f"{record:<6}{serial:>5} {atom:<4} {resname:<4}{chain}{resseq:>4}    "
        f"   0.000   0.000{3.0 * serial:8.3f}  1.00  0.00          {element:>2}  "
    )


def test_wide_resname_does_not_rename_an_atom_residue_of_the_same_number():
    lines = [
        _hetatm("ATOM", 1, "CA", "ALA", "A", 1, "C"),
        _hetatm("HETATM", 2, "C1", "ALAD", "A", 1, "C"),
    ]
    structure = PDBParser(QUIET=True, strict_parser=False).get_structure(
        io.StringIO("\n".join(lines) + "\nEND\n"), "atom-and-hetatm"
    )
    protein = [res for res in structure.get_residues() if res.id[0] == " "]
    assert [res.resname for res in protein] == ["ALA"]


def test_tip3_and_hoh_of_the_same_number_both_parse():
    lines = [
        _hetatm("HETATM", 1, "O", "HOH", "W", 1, "O"),
        _hetatm("HETATM", 2, "OH2", "TIP3", "W", 1, "O"),
    ]
    structure = PDBParser(QUIET=True, strict_parser=False).get_structure(
        io.StringIO("\n".join(lines) + "\nEND\n"), "two-waters"
    )
    residues = sorted((res.resname, res.id[0]) for res in structure.get_residues())
    assert residues == [("HOH", "W"), ("TIP3", "H_TIP3")]


def test_two_wide_resnames_of_the_same_number_both_restored():
    lines = [
        _hetatm("HETATM", 1, "P", "POPC", "L", 1, "P"),
        _hetatm("HETATM", 2, "O3", "CHOL", "L", 1, "O"),
    ]
    structure = PDBParser(QUIET=True, strict_parser=False).get_structure(
        io.StringIO("\n".join(lines) + "\nEND\n"), "two-lipids"
    )
    assert sorted(res.resname for res in structure.get_residues()) == ["CHOL", "POPC"]


@pytest.mark.parametrize("resname", ["POPC", "ALA"])
def test_parser_reads_an_open_file_handle(resname):
    # The SwissModel fetchers pass an io.StringIO and an explicit structure id.
    record = (
        f"HETATM    1  P   {resname:<4}L   1       0.000   0.000   0.000  1.00  0.00"
        "           P  "
    )
    handle = io.StringIO(f"{record}\nEND\n")
    structure = PDBParser(QUIET=True).get_structure(handle, "from-handle")
    residues = list(structure.get_residues())
    assert [res.resname for res in residues] == [resname]


def test_include_alt_writes_all_locations(load_structure):
    model = load_structure("2igd").models[0]

    def n_records(text):
        return sum(line.startswith(("ATOM", "HETATM")) for line in text.splitlines())

    assert n_records(get_pdb_str(model)) == 574
    assert n_records(get_pdb_str(model, include_alt=True)) == 606
