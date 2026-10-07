"""Solvation and ion placement on small systems."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from crimm.Modeller.Solvator import Solvator

CUTOFF = 6.0
SOLVCUT = 2.10
BULK_WATER_DENSITY = 0.0334  # molecules per cubic angstrom at 298 K


def solute_chains(model):
    return [chain for chain in model if chain.chain_type not in ("Solvent", "Ion")]


def heavy_coords(chains):
    return np.array(
        [a.coord for c in chains for a in c.get_atoms() if a.element != "H"],
        dtype=float,
    )


def chirality_signs(chain):
    """Sign of the N, C, CB arrangement around each CA; positive for L-amino acids."""
    signs = []
    for residue in chain:
        if not all(name in residue for name in ("N", "CA", "C", "CB")):
            continue
        n, ca, c, cb = (residue[name].coord for name in ("N", "CA", "C", "CB"))
        signs.append(np.sign(np.dot(np.cross(n - ca, c - ca), cb - ca)))
    return np.array(signs)


@pytest.fixture
def solvated_cube(prepared_model):
    model = prepared_model("1ubq")
    solvator = Solvator(model)
    water_chains = solvator.solvate(cutoff=CUTOFF, solvcut=SOLVCUT, box_type="cube")
    return model, solvator, water_chains


def test_cube_box_dimensions(solvated_cube):
    model, solvator, _ = solvated_cube
    extent = np.ptp(heavy_coords(solute_chains(model)), axis=0)
    assert solvator.box_dim >= extent.max() + 2 * CUTOFF - 1e-3
    info = model._solvation_info
    assert info["box_type"] == "cube"
    assert info["charmm_name"] == "CUBI"
    assert info["angles"] == (90.0, 90.0, 90.0)
    assert info["box_dims"] == pytest.approx([solvator.box_dim] * 3)


def test_waters_are_added_to_the_model_in_place(solvated_cube):
    model, _, water_chains = solvated_cube
    assert len(water_chains) >= 1
    for chain in water_chains:
        assert chain.chain_type == "Solvent"
        assert chain in list(model)
        for water in chain:
            assert sorted(atom.name for atom in water) == ["H1", "H2", "OH2"]


def test_waters_fill_the_box_at_bulk_density(solvated_cube):
    model, solvator, water_chains = solvated_cube
    n_water = sum(len(chain) for chain in water_chains)
    oxygens = np.array(
        [w["OH2"].coord for chain in water_chains for w in chain], dtype=float
    )
    half = solvator.box_dim / 2
    assert np.abs(oxygens).max() <= half + 1e-3
    # the solute displaces some water, so the box-wide density is a little below bulk
    density = n_water / solvator.box_dim**3
    assert 0.85 * BULK_WATER_DENSITY < density < BULK_WATER_DENSITY


def test_no_water_overlaps_the_solute(solvated_cube):
    model, _, water_chains = solvated_cube
    oxygens = np.array(
        [w["OH2"].coord for chain in water_chains for w in chain], dtype=float
    )
    solute = np.array(
        [a.coord for c in solute_chains(model) for a in c.get_atoms()], dtype=float
    )
    nearest, _ = cKDTree(solute).query(oxygens)
    assert nearest.min() >= SOLVCUT - 1e-3


def test_solute_is_centred_and_rigid(prepared_model):
    model = prepared_model("1ubq")
    protein = model.protein[0]
    before = np.array([atom.coord for atom in protein.get_atoms()], dtype=float)
    Solvator(model).solvate(cutoff=CUTOFF)
    after = np.array([atom.coord for atom in protein.get_atoms()], dtype=float)
    # internal geometry is unchanged
    sample = slice(0, 300, 3)
    d_before = np.linalg.norm(before[sample, None] - before[None, sample], axis=-1)
    d_after = np.linalg.norm(after[sample, None] - after[None, sample], axis=-1)
    assert d_after == pytest.approx(d_before, abs=1e-2)
    assert np.abs(after.mean(axis=0)).max() < 3.0


@pytest.mark.xfail(
    strict=True,
    reason="Default solvation of 1UBQ returns a mirror image: orient_coords_octa "
    "applies a reflection (docs/dev/ROADMAP.md, 1.1)",
)
def test_solvation_preserves_chirality(prepared_model):
    model = prepared_model("1ubq")
    protein = model.protein[0]
    before = chirality_signs(protein)
    Solvator(model).solvate(cutoff=CUTOFF, box_type="cube")
    after = chirality_signs(protein)
    assert len(before) > 60
    assert (after == before).all()


def test_existing_waters_removed_by_default(prepared_model):
    model = prepared_model("1ubq")
    # keep the residue objects alive so identity comparison is meaningful
    crystal_waters = [res for chain in model.solvent for res in chain]
    Solvator(model).solvate(cutoff=CUTOFF)
    remaining = {id(res) for chain in model.solvent for res in chain}
    assert not any(id(res) in remaining for res in crystal_waters)
    assert model._solvation_info["preserved_waters"] == 0


def test_no_orientation_keeps_coordinates(prepared_model):
    model = prepared_model("1crn")
    before = np.array([a.coord for a in model.protein[0].get_atoms()], dtype=float)
    Solvator(model).solvate(cutoff=CUTOFF, orient_coords=False)
    after = np.array([a.coord for a in model.protein[0].get_atoms()], dtype=float)
    shift = after - before
    # at most a rigid translation to the box centre, no rotation
    assert shift == pytest.approx(np.tile(shift[0], (len(shift), 1)), abs=1e-3)


def system_charge(model):
    return sum(
        atom.topo_definition.charge
        for atom in model.get_atoms()
        if atom.topo_definition is not None
    )


def ion_counts(model):
    counts = {}
    for chain in model:
        if chain.chain_type == "Ion":
            for residue in chain:
                counts[residue.resname] = counts.get(residue.resname, 0) + 1
    return counts


def test_ions_for_neutral_protein(prepared_model, capsys):
    model = prepared_model("1ubq")
    solvator = Solvator(model)
    solvator.solvate(cutoff=CUTOFF)
    n_water = sum(len(chain) for chain in model.solvent)
    ion_chain = solvator.add_ions(concentration=0.15)
    capsys.readouterr()  # add_ions prints a report
    counts = ion_counts(model)
    assert counts["SOD"] == counts["CLA"]
    assert ion_chain.chain_type == "Ion"
    # 0.15 M in pure water is one ion pair per 55.5 / 0.15 = 370 waters
    assert counts["SOD"] == pytest.approx(n_water * 0.15 / 55.5, abs=2)


def test_ions_neutralize_charged_dna(prepared_model, capsys):
    model = prepared_model("1bna")
    solvator = Solvator(model)
    solvator.solvate(cutoff=CUTOFF)
    solvator.add_ions(concentration=0.15)
    capsys.readouterr()
    counts = ion_counts(model)
    # two strands of -12 e each
    assert counts["SOD"] - counts["CLA"] == 24
    assert counts["CLA"] > 0


def test_ions_do_not_overlap_solute(prepared_model, capsys):
    model = prepared_model("1ubq")
    solvator = Solvator(model)
    solvator.solvate(cutoff=CUTOFF)
    solvator.add_ions(concentration=0.15, min_dist_solute=5.0)
    capsys.readouterr()
    ions = np.array(
        [a.coord for c in model if c.chain_type == "Ion" for a in c.get_atoms()],
        dtype=float,
    )
    solute = np.array(
        [a.coord for c in solute_chains(model) for a in c.get_atoms()], dtype=float
    )
    nearest, _ = cKDTree(solute).query(ions)
    assert nearest.min() >= 5.0 - 1e-3


def test_empty_model_is_rejected(prepared_model):
    model = prepared_model("1crn")
    model.remove_chains([chain.id for chain in list(model)])
    with pytest.raises(ValueError):
        Solvator(model).solvate()
