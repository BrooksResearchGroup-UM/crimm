"""The whole preparation pipeline, through pyCHARMM: protein, CGenFF ligand, ion, solvent.

``test_pycharmm.py`` checks protein-only models against what CHARMM generates itself. This
module runs the pipeline a user runs, on 3PTB (trypsin with benzamidine, a calcium ion,
crystal waters and six disulfides), and checks each stage on the way:

1. topology for every atom, the ligand typed by CGenFF;
2. PropKa protonation keeps every chain at an integer charge;
3. CHARMM loads the model with the same atoms and charge as crimm's;
4. restrained steepest descent in CHARMM, coordinates fetched back into crimm;
5. solvation and balancing ions, loaded into CHARMM as one neutral system;
6. steepest descent of the solvent with periodic images and the solute fixed;
7. the PSF that CHARMM writes reads back into crimm.

Loop building is not run: 3PTB has no gaps, and ``benchmarks/benchmark_pipeline.py`` covers
it. Needs pyCHARMM (``CHARMM_LIB_DIR``), the cgenff executable (``CRIMM_CGENFF_PATH``) and
the network (the CGenFF path queries RCSB for the ligand). Run with
``pytest -m "pycharmm and cgenff"``. Never call CHARMM's ``stop`` here: pyCHARMM dumps core.
"""

import glob
import os
import warnings

import numpy as np
import pytest
from conftest import parse_structure

from crimm.IO import read_psf
from crimm.Modeller import TopologyGenerator
from crimm.StructEntities.OrganizedModel import OrganizedModel

pytestmark = [
    pytest.mark.pycharmm,
    pytest.mark.cgenff,
    pytest.mark.network,
    pytest.mark.slow,
]

PDB_ID = "3ptb"
SD_STEPS = 100


def nonbonded(pycharmm, **cutoffs):
    return pycharmm.NonBondedScript(
        eps=1.0,
        cdie=True,
        atom=True,
        vatom=True,
        fswitch=True,
        vfswitch=True,
        **cutoffs,
    )


def total_energy():
    import pycharmm.energy as energy

    energy.show()
    return energy.get_total()


def positions():
    import pycharmm.coor as coor

    return coor.get_positions().to_numpy()


def integral(value):
    return value == pytest.approx(round(value), abs=1e-3)


@pytest.fixture(scope="module")
def pipeline(cgenff_path, tmp_path_factory):
    """Run every stage once and record what each test checks.

    pyCHARMM holds one PSF per process, so the stages share it and run in order here; the
    tests only read the record.
    """
    lib_dir = os.environ.get("CHARMM_LIB_DIR", "")
    if not lib_dir or not glob.glob(os.path.join(lib_dir, "libchmm.*")):
        pytest.skip("CHARMM library not found (set CHARMM_LIB_DIR)")
    pycharmm = pytest.importorskip("pycharmm")
    import pycharmm.psf as psf
    import pycharmm.write as write
    from pycharmm import crystal, image

    from crimm.Adaptors import pyCHARMMAdaptors as pca
    from crimm.Adaptors.PropKaAdaptors import PropKaProtonator
    from crimm.Modeller.Solvator import Solvator

    # Water hydrogens and ion placement use the global NumPy generator (G15).
    np.random.seed(0)
    out = tmp_path_factory.mktemp("pipeline")
    rec = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OrganizedModel(parse_structure(PDB_ID).models[0])

        # 1. topology, ligand through CGenFF
        topo = TopologyGenerator(cgenff_excutable_path=cgenff_path)
        topo.generate_model(model, coerce=True, QUIET=True)
        rec["untyped"] = [
            a for a in model.get_atoms() if getattr(a, "topo_definition", None) is None
        ]
        rec["ligand_residues"] = [res for chain in model.ligand for res in chain]

        # 2. protonation
        protonator = PropKaProtonator(topo, pH=7.4)
        protonator.load_model(model)
        protonator.apply_patches()
        rec["chain_charges"] = {chain.id: chain.total_charge for chain in model}

        # 3. load into CHARMM
        pca.empty_charmm()
        pca.load_model(model)
        rec["natom_dry"] = (psf.get_natom(), len(list(model.get_atoms())))
        rec["charge_dry"] = (sum(psf.get_charges()), sum(rec["chain_charges"].values()))

        # 4. restrained minimization, coordinates back into crimm
        ca = pycharmm.SelectAtoms(atom_type="CA")
        ca_idx = np.array(ca.get_atom_indexes())
        script = nonbonded(pycharmm, cutnb=18.0, ctonnb=13.0, ctofnb=17.0)
        script.run()
        before_e, before_ca = total_energy(), positions()[ca_idx]
        pca.sd_minimize(SD_STEPS, script, cons_harm_selection=ca)
        after_e, after_ca = total_energy(), positions()[ca_idx]
        rec["restrained_sd"] = (before_e, after_e)
        rec["ca_rmsd"] = float(
            np.sqrt(((after_ca - before_ca) ** 2).sum(axis=1).mean())
        )
        pca.fetch_coords_from_charmm(model)
        model_ca = np.array(
            [res["CA"].coord for chain in model.protein for res in chain if "CA" in res]
        )
        rec["ca_fetched"] = (model_ca, after_ca)

        # 5. solvate, add ions, load the solvated system
        solvator = Solvator(model)
        solvator.solvate(
            cutoff=8.0, solvcut=2.1, remove_existing_water=False, orient_coords=False
        )
        solvator.add_balancing_ions()
        pca.empty_charmm()
        pca.load_model(model)
        rec["natom_wet"] = (psf.get_natom(), len(list(model.get_atoms())))
        rec["charge_wet"] = sum(psf.get_charges())

        # 6. minimize the solvent under periodic boundaries, solute fixed
        solute_segids = {
            res.segid
            for chain in model
            if chain.chain_type not in ("Solvent", "Ion")
            for res in chain
        }
        box = solvator.box_dim
        crystal.define_cubic(box)
        crystal.build(box / 2)
        for segid in solute_segids:
            image.setup_segment(0.0, 0.0, 0.0, segid)
        image.setup_residue(0.0, 0.0, 0.0, "TIP3")
        for ion in {res.resname for chain in model.ion for res in chain}:
            image.setup_residue(0.0, 0.0, 0.0, ion)
        cutnb = min(box / 2, 12.0)
        script = nonbonded(
            pycharmm,
            cutnb=cutnb,
            cutim=cutnb,
            ctonnb=cutnb - 3.0,
            ctofnb=cutnb - 1.0,
            inbfrq=-1,
            imgfrq=-1,
        )
        solute = pycharmm.SelectAtoms()
        for segid in solute_segids:
            solute |= pycharmm.SelectAtoms(seg_id=segid)
        solute_idx = np.array(solute.get_atom_indexes())
        script.run()
        before_e, before_pos = total_energy(), positions()
        pca.sd_minimize(SD_STEPS, script, cons_fix_selection=solute)
        after_e, after_pos = total_energy(), positions()
        rec["pbc_sd"] = (before_e, after_e)
        rec["solute_shift"] = float(
            np.abs(after_pos[solute_idx] - before_pos[solute_idx]).max()
        )

        # 7. CHARMM writes the system; crimm reads it back
        write.psf_card(str(out / "system.psf"))
        write.coor_card(str(out / "system.crd"))
        rec["psf_read_back"] = read_psf(str(out / "system.psf"))

    yield rec
    # TODO(#129): empty_charmm() deletes atoms with image centering still on; CHARMM's
    # warning for that is fatal at the default BOMLEV and ends the process, so free the
    # crystal first. Drop this call when #129 is fixed.
    crystal.free()
    pca.empty_charmm()


def test_every_atom_is_typed_and_the_ligand_by_cgenff(pipeline):
    assert pipeline["untyped"] == []
    assert pipeline["ligand_residues"]
    assert all(res.topo_definition is not None for res in pipeline["ligand_residues"])


def test_protonation_keeps_chain_charges_integral(pipeline):
    charges = pipeline["chain_charges"]
    assert all(q is not None and integral(q) for q in charges.values()), charges


def test_charmm_loads_the_same_atoms_and_charge(pipeline):
    charmm_natom, crimm_natom = pipeline["natom_dry"]
    assert charmm_natom == crimm_natom
    charmm_q, crimm_q = pipeline["charge_dry"]
    assert charmm_q == pytest.approx(crimm_q, abs=1e-3)


def test_restrained_minimization_lowers_energy_and_holds_ca(pipeline):
    before, after = pipeline["restrained_sd"]
    assert np.isfinite(after)
    assert after < before
    assert pipeline["ca_rmsd"] < 0.5


def test_minimized_coordinates_are_fetched_into_crimm(pipeline):
    model_ca, charmm_ca = pipeline["ca_fetched"]
    np.testing.assert_allclose(model_ca, charmm_ca, atol=1e-4)


def test_solvated_system_loads_whole_and_neutral(pipeline):
    charmm_natom, crimm_natom = pipeline["natom_wet"]
    assert charmm_natom == crimm_natom > pipeline["natom_dry"][1]
    assert pipeline["charge_wet"] == pytest.approx(0.0, abs=1e-3)


def test_pbc_minimization_moves_only_the_solvent(pipeline):
    before, after = pipeline["pbc_sd"]
    assert np.isfinite(after)
    assert after < before
    assert pipeline["solute_shift"] == 0.0


def test_charmm_written_psf_reads_back(pipeline):
    assert len(pipeline["psf_read_back"].atoms) == pipeline["natom_wet"][0]
