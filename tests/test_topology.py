"""CHARMM topology generation on the fixture structures (no CGenFF)."""
import os
import warnings

import numpy as np
import pytest

from conftest import ALL_IDS, EXPECTED
from crimm.Modeller import TopologyGenerator

POLYMER_TYPES = ("Polypeptide(L)", "Polydeoxyribonucleotide", "Polyribonucleotide")


def polymers(model):
    return [chain for chain in model if chain.chain_type in POLYMER_TYPES]


@pytest.mark.parametrize("pdb_id", ALL_IDS)
def test_every_polymer_atom_is_typed(shared_prepared, pdb_id):
    for chain in polymers(shared_prepared(pdb_id)):
        assert chain.undefined_res == []
        assert all(atom.topo_definition is not None for atom in chain.get_atoms())


@pytest.mark.parametrize("pdb_id", ALL_IDS)
def test_polymer_charge_is_integer_and_expected(shared_prepared, pdb_id):
    model = shared_prepared(pdb_id)
    for chain in polymers(model):
        charge = chain.total_charge
        assert charge == pytest.approx(round(charge), abs=1e-6)
        assert round(charge) == EXPECTED[pdb_id]["polymer_charge"][chain.id]


@pytest.mark.parametrize("pdb_id", ALL_IDS)
def test_hydrogens_and_missing_atoms_are_built(shared_prepared, pdb_id):
    for chain in polymers(shared_prepared(pdb_id)):
        atoms = list(chain.get_atoms())
        coords = np.array([atom.coord for atom in atoms], dtype=float)
        assert np.isfinite(coords).all()
        assert any(atom.element == "H" for atom in atoms)


@pytest.mark.parametrize("pdb_id", ALL_IDS)
def test_bond_lengths_are_physical(shared_prepared, pdb_id):
    """Every bonded pair, including built hydrogens, sits at a covalent distance."""
    model = shared_prepared(pdb_id)
    lengths = np.array(
        [np.linalg.norm(a.coord - b.coord) for a, b in model.topology.bonds]
    )
    assert len(lengths) > 0
    assert lengths.min() > 0.8
    assert lengths.max() < 2.3


def test_water_gets_tip3_hydrogens(shared_prepared):
    solvent = shared_prepared("1ubq").solvent[0]
    assert len(solvent) == 58
    for water in solvent:
        assert sorted(atom.name for atom in water) == ["H1", "H2", "OH2"]
    assert solvent.total_charge == pytest.approx(0.0, abs=1e-6)


def test_disulfides_are_patched(shared_prepared):
    for pdb_id, n_bonds in (("1crn", 3), ("3ptb", 6), ("1ubq", 0)):
        model = shared_prepared(pdb_id)
        disulfides = model.topology.disulfide_topology
        assert len(disulfides.bonds) == n_bonds
        for sg1, sg2 in disulfides.bonds:
            assert (sg1.name, sg2.name) == ("SG", "SG")
            # the thiol hydrogen is removed from both bonded cysteines
            assert "HG1" not in sg1.get_parent()
            assert "HG1" not in sg2.get_parent()


def test_ion_is_typed_and_charged(shared_prepared):
    ion_chain = shared_prepared("3ptb").ion[0]
    assert ion_chain.total_charge == pytest.approx(2.0)


def test_terminal_patches_are_default_caps(shared_prepared):
    chain = shared_prepared("1ubq").protein[0]
    first, last = chain.residues[0], chain.residues[-1]
    # ACE cap on the N-terminus, CT3 (N-methylamide) on the C-terminus
    assert {"CAY", "CY", "OY"} <= {atom.name for atom in first}
    assert {"NT", "CAT"} <= {atom.name for atom in last}


def test_ligand_without_cgenff_warns_and_stays_untyped(organized_model):
    model = organized_model("3ptb")
    with pytest.warns(UserWarning, match="CGENFF is not configured"):
        TopologyGenerator().generate_model(model)
    ligand_atoms = list(model.ligand[0].get_atoms())
    assert all(atom.topo_definition is None for atom in ligand_atoms)


def test_discontinuous_chain_is_rejected(organized_model):
    chain = organized_model("1ubq").protein[0]
    for resseq in (30, 31, 32):
        chain.detach_child(chain[resseq].id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="discontinuous backbone"):
            TopologyGenerator().generate(chain, QUIET=True)


def test_generate_model_rejects_plain_model(load_structure):
    with pytest.raises(TypeError):
        TopologyGenerator().generate_model(load_structure("1crn").models[0])


@pytest.mark.xfail(
    strict=True,
    reason="Dihedral.angle is a stub that returns 0.0 for every dihedral",
)
def test_dihedral_angle_is_computed(shared_prepared):
    dihedrals = shared_prepared("1crn").topology.dihedrals
    assert any(abs(dihedral.angle) > 1.0 for dihedral in dihedrals[:200])


@pytest.mark.cgenff
@pytest.mark.network
def test_ligand_topology_with_cgenff(organized_model, cgenff_path, tmp_path):
    # network: the ligand's bond orders and hydrogens are looked up on RCSB
    model = organized_model("3ptb")
    generator = TopologyGenerator(
        cgenff_excutable_path=cgenff_path, cgenff_output_path=str(tmp_path)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.generate_model(model, QUIET=True)
    ligand_atoms = list(model.ligand[0].get_atoms())
    assert all(atom.topo_definition is not None for atom in ligand_atoms)
    charge = sum(atom.topo_definition.charge for atom in ligand_atoms)
    assert charge == pytest.approx(round(charge), abs=1e-4)
    assert os.listdir(tmp_path)
