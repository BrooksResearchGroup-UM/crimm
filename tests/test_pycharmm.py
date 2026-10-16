"""crimm and pyCHARMM: what crimm builds is what CHARMM builds.

Needs pyCHARMM and a CHARMM shared library found through ``CHARMM_LIB_DIR``; every test is
marked ``pycharmm``, so the default run deselects them and CI never runs them. CHARMM is
licensed (see CONTRIBUTING). Run with ``pytest -m pycharmm``.

The reference is CHARMM generating the same segments itself from the RTF: same sequence,
same terminal patches, ``patch DISU`` for each disulfide, and crimm's coordinates. Models are
protein only (no solvent), so no water or ion segments need building on the CHARMM side.

pyCHARMM holds one PSF per process, so every test starts by deleting all atoms. Never call
CHARMM's ``stop`` from here: pyCHARMM dumps core on it.
"""
import glob
import os
import warnings
from collections import Counter

import numpy as np
import pytest
from conftest import parse_structure

from crimm.IO import read_psf, write_crd
from crimm.Modeller import TopologyGenerator
from crimm.StructEntities.OrganizedModel import OrganizedModel
from crimm.Utils.StructureUtils import polymer_chain_to_charmm_segid

pytestmark = pytest.mark.pycharmm

ENERGY_TERMS = ("BOND", "ANGL", "UREY", "DIHE", "IMPR", "CMAP", "VDW", "ELEC")

DISULFIDE_DIHEDRALS = pytest.mark.xfail(
    strict=True,
    reason="#112: disulfide cysteines get N/C-CA-CB-SG twice and lack HB-CB-SG-SG",
)

# (pdb_id, C-terminal patch). 1CRN uses CTER: the default CT3 cap is built onto the
# Arg10-Asn46 salt bridge (docs/dev/PDB_PREP_LOG.md). None keeps the default.
CASES = [("1crn", "CTER"), ("1ubq", None), ("2igd", None)]
CASES_VS_GENERATED = [
    pytest.param(*CASES[0], marks=DISULFIDE_DIHEDRALS),
    *CASES[1:],
]


@pytest.fixture(scope="module")
def pca():
    """crimm's pyCHARMM adaptor module; skips when CHARMM is not available."""
    lib_dir = os.environ.get("CHARMM_LIB_DIR", "")
    if not lib_dir or not glob.glob(os.path.join(lib_dir, "libchmm.*")):
        pytest.skip("CHARMM library not found (set CHARMM_LIB_DIR)")
    pytest.importorskip("pycharmm")
    from crimm.Adaptors import pyCHARMMAdaptors

    return pyCHARMMAdaptors


@pytest.fixture
def fresh_charmm(pca):
    """The adaptor, with no atoms left in CHARMM from an earlier test."""
    pca.empty_charmm()
    yield pca
    pca.empty_charmm()


def protein_model(pdb_id, last_patch):
    model = OrganizedModel(parse_structure(pdb_id, include_solvent=False).models[0])
    options = {} if last_patch is None else {"prot_last_patch": last_patch}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TopologyGenerator().generate_model(model, QUIET=True, **options)
    return model


def load(pca, model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca.load_model(model)


def generate_in_charmm(model):
    """Replace the PSF in CHARMM with segments CHARMM generates itself for ``model``."""
    import pycharmm.generate as generate
    import pycharmm.psf as psf
    import pycharmm.read as read

    from crimm.Adaptors.pyCHARMMAdaptors import patch_disu_from_model

    psf.delete_atoms()
    for chain in model.protein:
        read.sequence_string(" ".join(res.topo_definition.resname for res in chain))
        generate.new_segment(
            seg_name=polymer_chain_to_charmm_segid(chain),
            first_patch=chain.residues[0].topo_definition.patch_with,
            last_patch=chain.residues[-1].topo_definition.patch_with,
            setup_ic=True,
        )
    patch_disu_from_model(model)


def charmm_psf(path):
    """Write the PSF currently in CHARMM and return it keyed by atom (see ``keyed``)."""
    import pycharmm.write as write

    write.psf_card(str(path))
    return keyed(read_psf(str(path)))


def keyed(psf_data):
    """Atoms and bonded terms keyed by (segid, resid, atom name), independent of atom order.

    Terms are multisets, so a term listed twice differs from one listed once.
    """
    key = {a.serial: (a.segid, a.resid.strip(), a.atomname) for a in psf_data.atoms}
    atoms = {key[a.serial]: (a.atomtype, round(a.charge, 4)) for a in psf_data.atoms}

    def terms(items, symmetric=True):
        out = Counter()
        for item in items:
            k = tuple(key[i] for i in item)
            out[min(k, k[::-1]) if symmetric else k] += 1
        return out

    return {
        "atoms": atoms,
        "bonds": terms(psf_data.bonds),
        "angles": terms(psf_data.angles),
        "dihedrals": terms(psf_data.dihedrals),
        "impropers": terms(psf_data.impropers, symmetric=False),
    }


def energy_terms():
    import pycharmm.energy as energy

    energy.show()
    return {term: energy.get_term_by_name(term) for term in ENERGY_TERMS}


def assert_same_psf(actual, expected):
    assert actual["atoms"] == expected["atoms"]
    for name in ("bonds", "angles", "dihedrals", "impropers"):
        assert actual[name] == expected[name], name


@pytest.mark.parametrize("pdb_id, last_patch", CASES)
def test_charmm_reads_crimm_psf_as_written(fresh_charmm, tmp_path, pdb_id, last_patch):
    import pycharmm.psf as psf

    from crimm.IO import write_psf

    model = protein_model(pdb_id, last_patch)
    load(fresh_charmm, model)
    assert psf.get_natom() == len(list(model.get_atoms()))

    write_psf(model, str(tmp_path / "crimm.psf"))
    written = keyed(read_psf(str(tmp_path / "crimm.psf")))
    assert_same_psf(charmm_psf(tmp_path / "charmm.psf"), written)


@pytest.mark.parametrize("pdb_id, last_patch", CASES_VS_GENERATED)
def test_crimm_psf_matches_charmm_generated(fresh_charmm, tmp_path, pdb_id, last_patch):
    model = protein_model(pdb_id, last_patch)
    load(fresh_charmm, model)
    from_crimm = charmm_psf(tmp_path / "from_crimm.psf")

    generate_in_charmm(model)
    assert_same_psf(from_crimm, charmm_psf(tmp_path / "generated.psf"))


@pytest.mark.parametrize("pdb_id, last_patch", CASES_VS_GENERATED)
def test_energy_matches_charmm_generated(fresh_charmm, tmp_path, pdb_id, last_patch):
    import pycharmm.coor as coor
    import pycharmm.read as read

    model = protein_model(pdb_id, last_patch)
    load(fresh_charmm, model)
    from_crimm = energy_terms()

    crd_path = str(tmp_path / "model.crd")
    write_crd(model, crd_path)
    generate_in_charmm(model)
    read.coor_card(crd_path)
    # CHARMM leaves unset coordinates at 9999.0; every atom must have been read.
    assert np.abs(coor.get_positions().to_numpy()).max() < 9999.0
    generated = energy_terms()

    assert all(np.isfinite(value) for value in from_crimm.values())
    for term in ENERGY_TERMS:
        assert from_crimm[term] == pytest.approx(generated[term], abs=1e-3), term


def test_fetch_coords_from_charmm_moves_the_model(fresh_charmm):
    import pycharmm.coor as coor

    model = protein_model("1ubq", None)
    load(fresh_charmm, model)
    atoms = list(model.get_atoms())
    before = np.array([atom.coord for atom in atoms])

    shift = np.array([1.5, -2.0, 0.5])
    positions = coor.get_positions()
    coor.set_positions(positions + shift)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh_charmm.fetch_coords_from_charmm(model)

    after = np.array([atom.coord for atom in atoms])
    np.testing.assert_allclose(after - before, np.broadcast_to(shift, before.shape), atol=1e-4)
