"""CHARMM topology generation on the fixture structures (no CGenFF)."""

import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.vectors import calc_dihedral
from conftest import ALL_IDS, EXPECTED

from crimm.Modeller import TopologyGenerator
from crimm.StructEntities.TopoElements import CMap, Dihedral, Improper

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


def _angle_difference(a, b):
    """a - b in degrees, wrapped into [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def _stub_atom(name, coord):
    """The few Atom attributes a TopoEntity reads, with a given coordinate."""
    return SimpleNamespace(
        name=name,
        coord=np.array(coord, dtype=np.float32),
        topo_definition=SimpleNamespace(atom_type=name),
        get_full_id=lambda: ("A", (" ", 1, " "), name),
    )


@pytest.mark.parametrize("element", [Dihedral, Improper])
@pytest.mark.parametrize(
    "expected", [0.0, 60.0, -60.0, 120.0, -120.0, 180.0, 179.9, -179.9]
)
def test_dihedral_angle_of_constructed_geometry(element, expected):
    # Looking along b->c, d is turned counterclockwise from a by `expected` when
    # seen from c, so the IUPAC dihedral (positive = clockwise from b) is `expected`.
    r = np.radians(expected)
    coords = [(1.0, 0.0, 0.0), (0, 0, 0), (0, 0, 1.5), (np.cos(r), np.sin(r), 1.5)]
    atoms = [_stub_atom(n, c) for n, c in zip("ABCD", coords)]
    angle = element(*atoms).angle
    assert isinstance(angle, float)
    assert -180.0 < angle <= 180.0
    assert _angle_difference(angle, expected) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("element", [Dihedral, Improper])
def test_dihedral_angle_is_none_when_an_atom_has_no_coordinates(element):
    atoms = [_stub_atom(n, (i, i * i, 0)) for i, n in enumerate("ABCD")]
    atoms[2].coord = None
    assert element(*atoms).angle is None


@pytest.mark.xfail(
    strict=True,
    reason="#166: CMap.angle unpacks four coordinates from its two Dihedrals and raises",
)
def test_cmap_repr_does_not_raise():
    coords = [(1.0, 0.0, 0.0), (0, 0, 0), (0, 0, 1.5), (0, 1.0, 1.5), (1.0, 1.0, 2.0)]
    atoms = [_stub_atom(n, c) for n, c in zip("ABCDE", coords)]
    cmap = CMap(Dihedral(*atoms[:4]), Dihedral(*atoms[1:]))
    repr(cmap)


def test_dihedral_angle_is_computed(shared_prepared):
    dihedrals = shared_prepared("1crn").topology.dihedrals
    assert any(abs(dihedral.angle) > 1.0 for dihedral in dihedrals[:200])


def test_dihedral_and_improper_angles_match_biopython(shared_prepared):
    topology = shared_prepared("1crn").topology
    for element in list(topology.dihedrals) + list(topology.impropers):
        expected = np.degrees(calc_dihedral(*(atom.get_vector() for atom in element)))
        assert _angle_difference(element.angle, expected) == pytest.approx(
            0.0, abs=1e-3
        ), element


def test_backbone_phi_psi_match_biopython(shared_prepared):
    """phi = C(i-1)-N-CA-C and psi = N-CA-C-N(i+1), against Bio.PDB's polypeptide."""
    chain = shared_prepared("1crn").protein[0]
    by_names = {}
    for dihedral in chain.topology.dihedrals:
        key = tuple((atom.get_parent().id[1], atom.name) for atom in dihedral)
        by_names[key] = by_names[key[::-1]] = dihedral
    residues = chain.residues
    reference = PPBuilder().build_peptides(chain)[0].get_phi_psi_list()
    assert len(reference) == len(residues)
    n_checked = 0
    for prev, res, nxt, (phi, psi) in zip(
        residues[:-2], residues[1:-1], residues[2:], reference[1:-1]
    ):
        i = res.id[1]
        phi_key = ((prev.id[1], "C"), (i, "N"), (i, "CA"), (i, "C"))
        psi_key = ((i, "N"), (i, "CA"), (i, "C"), (nxt.id[1], "N"))
        for key, expected in ((phi_key, phi), (psi_key, psi)):
            difference = _angle_difference(by_names[key].angle, np.degrees(expected))
            assert difference == pytest.approx(0.0, abs=1e-3), key
        n_checked += 1
    assert n_checked == len(residues) - 2


DISULFIDE_DIHEDRALS = pytest.mark.xfail(
    strict=True,
    reason="#112: disulfide cysteines get N/C-CA-CB-SG twice and lack HB-CB-SG-SG",
)


@pytest.mark.parametrize(
    "pdb_id",
    [
        pytest.param(pdb_id, marks=DISULFIDE_DIHEDRALS)
        if pdb_id in ("1crn", "3ptb")
        else pdb_id
        for pdb_id in ALL_IDS
    ],
)
def test_no_dihedral_is_listed_twice(shared_prepared, pdb_id):
    seen = set()
    duplicated = []
    for dihedral in shared_prepared(pdb_id).topology.dihedrals:
        key = tuple((a.parent.parent.id, a.parent.id[1], a.name) for a in dihedral)
        key = min(key, key[::-1])
        if key in seen:
            duplicated.append(key)
        seen.add(key)
    assert duplicated == []


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
