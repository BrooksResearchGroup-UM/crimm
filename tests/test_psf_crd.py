"""PSF and CRD writing, reading, and round trips."""
import warnings

import numpy as np
import pytest

from crimm.IO import (
    CRDParser,
    get_crd_str,
    get_psf_str,
    read_psf,
    write_crd,
    write_psf,
)
from crimm.Modeller import TopologyGenerator

# Fixtures whose PSF and CRD agree today. The others are covered by the xfail tests below.
CONSISTENT = ["1ubq", "1crn", "2igd"]


@pytest.fixture
def written(shared_prepared, tmp_path):
    """Factory: write PSF and CRD for a fixture, return (model, psf_path, crd_path)."""

    def write(pdb_id):
        model = shared_prepared(pdb_id)
        psf_path = tmp_path / f"{pdb_id}.psf"
        crd_path = tmp_path / f"{pdb_id}.crd"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_psf(model, str(psf_path))
            write_crd(model, str(crd_path))
        return model, str(psf_path), str(crd_path)

    return write


def crd_atoms(crd_path):
    return list(CRDParser().get_structure(crd_path).get_atoms())


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_psf_matches_model(written, pdb_id):
    model, psf_path, _ = written(pdb_id)
    psf = read_psf(psf_path)
    topology = model.topology
    assert len(psf.atoms) == len(list(model.get_atoms()))
    assert len(psf.bonds) == len(topology.bonds)
    assert len(psf.angles) == len(topology.angles)
    assert len(psf.dihedrals) == len(topology.dihedrals)
    assert len(psf.impropers) == len(topology.impropers)
    assert psf.extended and psf.xplor


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_psf_charge_matches_model(written, pdb_id):
    model, psf_path, _ = written(pdb_id)
    psf_charge = sum(atom.charge for atom in read_psf(psf_path).atoms)
    model_charge = sum(atom.topo_definition.charge for atom in model.get_atoms())
    assert psf_charge == pytest.approx(model_charge, abs=1e-4)
    assert psf_charge == pytest.approx(round(psf_charge), abs=1e-4)


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_psf_and_crd_have_the_same_atoms(written, pdb_id):
    _, psf_path, crd_path = written(pdb_id)
    assert len(read_psf(psf_path).atoms) == len(crd_atoms(crd_path))


def test_psf_segids(written):
    _, psf_path, _ = written("1ubq")
    assert {atom.segid for atom in read_psf(psf_path).atoms} == {"PROA", "SOLV"}


def test_protein_psf_has_one_cmap_per_residue(written):
    _, psf_path, _ = written("1ubq")
    psf = read_psf(psf_path)
    assert psf.has_cmap
    assert len(psf.cmap) == 76


def test_bond_indices_are_valid(written):
    _, psf_path, _ = written("1crn")
    psf = read_psf(psf_path)
    n_atoms = len(psf.atoms)
    for i, j in psf.bonds:
        assert 1 <= i <= n_atoms and 1 <= j <= n_atoms and i != j


def test_string_and_file_output_agree(shared_prepared, tmp_path):
    """Apart from the title block (timestamp, user), both routes give the same text."""
    model = shared_prepared("1crn")
    path = tmp_path / "out.crd"
    write_crd(model, str(path))

    def body(text):
        return [line for line in text.splitlines() if not line.startswith("*")]

    assert body(path.read_text()) == body(get_crd_str(model))
    assert "PSF" in get_psf_str(model).splitlines()[0]


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_crd_round_trip_preserves_coordinates(written, pdb_id):
    model, _, crd_path = written(pdb_id)
    original = np.array([atom.coord for atom in model.get_atoms()], dtype=float)
    reread = np.array([atom.coord for atom in crd_atoms(crd_path)], dtype=float)
    assert reread.shape == original.shape
    # atom order differs between the model and the file, so compare as sets of points
    assert np.sort(reread, axis=0) == pytest.approx(np.sort(original, axis=0), abs=1e-5)


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_load_psf_crd_round_trip(written, pdb_id):
    model, psf_path, crd_path = written(pdb_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reloaded = TopologyGenerator().load_psf_crd(psf_path, crd_path, QUIET=True)
    assert type(reloaded).__name__ == "OrganizedModel"
    assert len(list(reloaded.get_atoms())) == len(list(model.get_atoms()))
    assert [len(chain) for chain in reloaded.protein] == [
        len(chain) for chain in model.protein
    ]


@pytest.mark.xfail(
    strict=True,
    reason="load_psf_crd leaves the terminal patch atoms (ACE and CT3 caps) without a "
    "topo_definition, so total_charge of the reloaded chain is None",
)
def test_load_psf_crd_restores_charges(written):
    model, psf_path, crd_path = written("1crn")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reloaded = TopologyGenerator().load_psf_crd(psf_path, crd_path, QUIET=True)
    old, new = model.protein[0], reloaded.protein[0]
    assert all(atom.topo_definition is not None for atom in new.get_atoms())
    assert new.total_charge == pytest.approx(old.total_charge, abs=1e-4)


def test_write_psf_reports_untyped_ligand(shared_prepared, tmp_path):
    messages = write_psf(shared_prepared("3ptb"), str(tmp_path / "t.psf"))
    assert any("BEN" in message and "No topology" in message for message in messages)


@pytest.mark.xfail(
    strict=True,
    reason="With an untyped ligand the PSF skips its atoms but the CRD writes them "
    "(3PTB: 3416 vs 3425), so the pair cannot be loaded together",
)
def test_psf_and_crd_agree_with_untyped_ligand(written):
    _, psf_path, crd_path = written("3ptb")
    assert len(read_psf(psf_path).atoms) == len(crd_atoms(crd_path))


@pytest.mark.xfail(
    strict=True,
    reason="PSFWriter writes 394 atoms per DNA strand where the model has 383 "
    "(one duplicated atom in 11 of 12 residues)",
)
def test_dna_psf_atom_count_matches_model(written):
    model, psf_path, _ = written("1bna")
    assert len(read_psf(psf_path).atoms) == len(list(model.get_atoms()))


@pytest.mark.xfail(
    strict=True,
    reason="DNA PSF total charge is +9 where the model's is -24",
)
def test_dna_psf_charge_matches_model(written):
    model, psf_path, _ = written("1bna")
    psf_charge = sum(atom.charge for atom in read_psf(psf_path).atoms)
    model_charge = sum(atom.topo_definition.charge for atom in model.get_atoms())
    assert psf_charge == pytest.approx(model_charge, abs=1e-4)
