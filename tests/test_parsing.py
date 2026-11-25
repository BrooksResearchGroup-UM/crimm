"""mmCIF parsing and the basic entity hierarchy."""

import numpy as np
import pytest
from conftest import ALL_IDS, EXPECTED


@pytest.mark.parametrize("pdb_id", ALL_IDS)
def test_chain_types_and_sizes(load_structure, pdb_id):
    model = load_structure(pdb_id).models[0]
    found = [(chain.id, chain.chain_type, len(chain)) for chain in model]
    assert found == EXPECTED[pdb_id]["chains"]


@pytest.mark.parametrize("pdb_id", ALL_IDS)
def test_atom_count_and_coordinates(load_structure, pdb_id):
    model = load_structure(pdb_id).models[0]
    atoms = list(model.get_atoms())
    assert len(atoms) == EXPECTED[pdb_id]["n_atoms"]
    coords = np.array([atom.coord for atom in atoms])
    assert coords.shape == (len(atoms), 3)
    assert np.isfinite(coords).all()
    # hydrogens were excluded by the parser options
    assert all(atom.element != "H" for atom in atoms)


def test_structure_metadata(load_structure):
    structure = load_structure("1ubq")
    assert structure.id == "1UBQ"
    assert structure.level == "S"
    assert len(structure.models) == 1
    assert float(structure.resolution) == pytest.approx(1.8)
    assert {"name", "keywords", "citation", "idcode", "deposition_date"} <= set(
        structure.header
    )
    assert structure.models[0].pdb_id == "1UBQ"


def test_hierarchy_levels_and_parents(load_structure):
    structure = load_structure("1ubq")
    model = structure.models[0]
    chain = model["A"]
    residue = chain[5]
    atom = residue["CA"]
    assert [e.level for e in (structure, model, chain, residue, atom)] == list("SMCRA")
    assert chain.parent is model
    assert model.parent is structure
    assert chain.get_top_parent() is structure


def test_residue_and_atom_content(load_structure):
    residue = load_structure("1ubq").models[0]["A"][5]
    assert residue.resname == "VAL"
    assert residue.id == (" ", 5, " ")
    assert [atom.name for atom in residue] == ["N", "CA", "C", "O", "CB", "CG1", "CG2"]
    ca = residue["CA"]
    assert ca.element == "C"
    assert ca.coord == pytest.approx([28.605, 33.965, 12.503], abs=1e-3)
    assert ca.occupancy == pytest.approx(1.0)


def test_exclude_solvent(load_structure):
    model = load_structure("1ubq", include_solvent=False).models[0]
    assert [(chain.id, len(chain)) for chain in model] == [("A", 76)]


def test_sequences_of_complete_protein(load_structure):
    chain = load_structure("1ubq").models[0]["A"]
    assert str(chain.can_seq).startswith("MQIFVKTLTGKTITLEVEPS")
    assert len(chain.can_seq) == 76
    assert str(chain.seq) == str(chain.can_seq)
    assert chain.reported_res[:3] == [(1, "MET"), (2, "GLN"), (3, "ILE")]
    assert chain.missing_res == []
    assert chain.gaps == []
    assert chain.is_continuous()


def test_removing_residues_creates_a_gap(load_structure):
    chain = load_structure("1ubq").models[0]["A"]
    for resseq in (30, 31, 32):
        chain.detach_child(chain[resseq].id)
    assert chain.missing_res == [(30, "ILE"), (31, "GLN"), (32, "ASP")]
    assert chain.gaps == [{30, 31, 32}]
    assert not chain.is_continuous()
    assert "---" in str(chain.masked_seq)
    assert len(chain) == 73


def test_dna_canonical_sequence(load_structure):
    model = load_structure("1bna").models[0]
    for chain_id in ("A", "B"):
        assert str(model[chain_id].can_seq) == "CGCGAATTCGCG"


def test_dna_present_sequence_matches_canonical(load_structure):
    chain = load_structure("1bna").models[0]["A"]
    assert str(chain.seq) == str(chain.can_seq)


def _polymer_chain(chain_type, resnames):
    from crimm.StructEntities.Atom import Atom
    from crimm.StructEntities.Chain import PolymerChain
    from crimm.StructEntities.Residue import Residue

    reported = list(enumerate(resnames, start=1))
    chain = PolymerChain("A", 1, "A", chain_type, "", "X" * len(resnames), reported)
    for resseq, resname in reported:
        residue = Residue((" ", resseq, " "), resname, " ")
        residue.add(Atom("P", np.zeros(3), 0.0, 1.0, " ", " P", resseq, "P"))
        chain.add(residue)
    return chain


@pytest.mark.parametrize(
    "chain_type, resnames, expected",
    [
        # DNA, RNA and modified nucleotides (PSU, 5MC) map to their parent base
        ("Polydeoxyribonucleotide", ["DA", "DC", "DG", "DT"], "ACGT"),
        ("Polyribonucleotide", ["A", "C", "G", "U", "PSU", "5MC"], "ACGUUC"),
        ("Polyribonucleotide", ["A", "UNL"], "AX"),
        ("Polypeptide(L)", ["ALA", "MSE", "UNK"], "AMX"),
    ],
)
def test_present_sequence_one_letter_codes(chain_type, resnames, expected):
    assert str(_polymer_chain(chain_type, resnames).seq) == expected


def test_sequence_from_reported_residues_for_dna():
    from crimm.IO.MMCIFParser import _one_letter_seq

    reported = [(1, "DC"), (2, "DG"), (3, "DA"), (4, "DT"), (5, "A"), (6, "U")]
    assert _one_letter_seq(reported, "polydeoxyribonucleotide") == "CGATAU"


@pytest.mark.xfail(
    strict=True,
    reason="#165: chain_to_polymerchain checks residue names against Biopython's padded "
    "nucleic keys ('DA '), so a DNA or RNA chain raises ValueError",
)
def test_chain_to_polymerchain_accepts_dna(load_structure):
    from crimm.Utils.StructureUtils import chain_to_polymerchain

    chain = load_structure("1bna").models[0]["A"]
    polymer = chain_to_polymerchain(chain)
    assert str(polymer.seq) == "CGCGAATTCGCG"


@pytest.mark.xfail(
    strict=True,
    reason="#165: PDBParser checks residue names against Biopython's padded nucleic keys "
    "('DA '), so a DNA chain read from PDB stays an untyped Chain",
)
def test_pdb_parser_types_dna_chain(load_structure, tmp_path):
    from crimm.IO import PDBParser
    from crimm.IO.PDBString import get_pdb_str

    path = tmp_path / "1bna.pdb"
    path.write_text(get_pdb_str(load_structure("1bna").models[0]))
    chain = PDBParser(QUIET=True).get_structure(str(path)).models[0]["A"]
    assert chain.chain_type != "Chain"


def test_connect_records(load_structure):
    assert len(load_structure("1crn").models[0].connect_dict["disulf"]) == 3
    trypsin = load_structure("3ptb").models[0].connect_dict
    assert len(trypsin["disulf"]) == 6
    assert len(trypsin["metalc"]) == 6
    first = load_structure("1crn").models[0].connect_dict["disulf"][0]
    assert [end["atom_id"] for end in first] == ["SG", "SG"]
    assert [end["resseq"] for end in first] == [3, 40]


def test_alternate_locations(load_structure):
    model = load_structure("2igd").models[0]
    selected = list(model.get_atoms())
    everything = list(model.get_atoms(include_alt=True))
    assert len(selected) == 574
    assert len(everything) == 606
    disordered = [atom for atom in selected if atom.is_disordered() == 2]
    assert len(disordered) == 37
    assert sorted(disordered[0].child_dict) == ["A", "B"]
    # the first altloc is the one selected by default
    assert disordered[0].altloc == "A"
