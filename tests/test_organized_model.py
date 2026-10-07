"""OrganizedModel: offline chain classification."""
import pytest

from crimm.StructEntities.OrganizedModel import OrganizedModel


def summary(chains):
    return [(chain.id, chain.chain_type, len(chain)) for chain in chains]


def test_protein_and_solvent(organized_model):
    model = organized_model("1ubq")
    assert summary(model.protein) == [("A", "Polypeptide(L)", 76)]
    assert summary(model.solvent) == [("B", "Solvent", 58)]
    assert model.ligand == [] and model.ion == []
    assert model.pdb_id == "1UBQ"


def test_ligand_and_ion_are_separated(organized_model):
    model = organized_model("3ptb")
    assert summary(model.protein) == [("A", "Polypeptide(L)", 223)]
    assert summary(model.ligand) == [("B", "Ligand", 1)]
    assert summary(model.ion) == [("C", "Ion", 1)]
    assert summary(model.solvent) == [("D", "Solvent", 62)]
    assert model.ligand[0].residues[0].resname == "BEN"
    assert [chain.id for chain in model.non_solvent] == ["A", "B", "C"]


def test_dna_chains_and_merged_solvent(organized_model):
    model = organized_model("1bna")
    assert [chain.id for chain in model.DNA] == ["A", "B"]
    assert model.RNA == [] and model.protein == []
    # the two crystallographic water chains are merged into one
    assert summary(model.solvent) == [("C", "Solvent", 80)]


def test_water_oxygen_renamed_for_charmm(organized_model):
    model = organized_model("1ubq")
    names = {atom.name for atom in model.solvent[0].get_atoms()}
    assert names == {"OH2"}


def test_water_oxygen_rename_can_be_disabled(load_structure):
    raw = load_structure("1ubq").models[0]
    model = OrganizedModel(raw, rename_solvent_oxygen=False)
    names = {atom.name for atom in model.solvent[0].get_atoms()}
    assert names == {"O"}


def test_atom_count_is_preserved(organized_model, load_structure):
    for pdb_id in ("1ubq", "3ptb", "1bna", "2igd"):
        raw = len(list(load_structure(pdb_id).models[0].get_atoms()))
        assert len(list(organized_model(pdb_id).get_atoms())) == raw


def test_filter_rejects_unknown_type(organized_model):
    with pytest.raises(KeyError):
        organized_model("1crn").filter("not_a_type")


def test_rejects_chain_level_entity(load_structure):
    chain = load_structure("1crn").models[0]["A"]
    with pytest.raises(ValueError):
        OrganizedModel(chain)


def test_heterogen_map_overrides_classification(load_structure):
    raw = load_structure("3ptb").models[0]
    model = OrganizedModel(raw, heterogen_map={"BEN": "CoSolvent"})
    assert model.ligand == []
    assert [chain.residues[0].resname for chain in model.co_solvent] == ["BEN"]
